use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{get_device, log_softmax, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use std::time::Instant;
use kdam::tqdm;
use rand::{Rng, SeedableRng};
use rand::distributions::WeightedIndex;
use rand::prelude::{Distribution, IteratorRandom};
use crate::services::algorithms::exports::model_based::alpha_zero::alpha_zero::AlphaZeroLogger;
// bring in only the underlying search kernel, not the CLI runner:

/// split a length-(A+1) tensor into ([0..A) policy logits, [A] value)
fn split_policy_value<B: Backend>(
    out: Tensor<B, 1>,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    // out.dims() returns &[usize]
    let dim = out.dims()[0];
    let policy_logits = out.clone().slice([0..(dim - 1)]);
    let value_v       = out.slice([(dim - 1)..dim]);
    (policy_logits, value_v)
}

/// run num_sims of UCT from `root_env` (cloned), then return a π proportional to child visits.
pub fn run_mcts_policy<
    const S: usize,
    const A: usize,
    Env: DeepDiscreteActionsEnv<S, A> + Clone,
    R: Rng + ?Sized,
>(
    root_env: &Env,
    num_sims: usize,
    c: f32,
    rng: &mut R,
) -> [f32; A] {
    // ─────────────  Node definition  ────────────────────────────────
    #[derive(Clone)]
    struct Node<const A: usize> {
        visits:   usize,
        value:    f32,
        children: [Option<usize>; A],
        untried:  Vec<usize>,
    }
    impl<const A: usize> Node<A> {
        fn new(avail: impl Iterator<Item = usize>) -> Self {
            Self {
                visits:   0,
                value:    0.0,
                children: [(); A].map(|_| None),
                untried:  avail.collect(),
            }
        }
    }

    // ─────────────  MCTS tree storage  ──────────────────────────────
    let mut tree:   Vec<Node<A>> = Vec::new();
    let mut states: Vec<Env>     = Vec::new();

    // root
    tree.push(Node::new(root_env.available_actions_ids()));
    states.push(root_env.clone());

    // ─────────────  Simulations  ────────────────────────────────────
    for _ in 0..num_sims {
        // 1) Selection
        let mut node_id = 0;
        let mut env     = root_env.clone();
        let mut path    = vec![0];

        // 1) Selection
        while tree[node_id].untried.is_empty() && !env.is_game_over() {
            let parent_n = tree[node_id].visits as f32;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_a = usize::MAX;

            // Get current legal moves
            let current_legal_moves: Vec<_> = env.available_actions_ids().collect();

            for a in 0..A {
                if let Some(child_id) = tree[node_id].children[a] {
                    // Only consider moves that are currently legal
                    if current_legal_moves.contains(&a) {
                        let child = &tree[child_id];
                        let q = child.value / child.visits as f32; // mean value
                        let u = q + c * (parent_n.ln() / (child.visits as f32 + 1e-8)).sqrt();
                        if u > best_score {
                            best_score = u;
                            best_a = a;
                        }
                    }
                }
            }

            // If we never found a legal child, break out to expansion
            if best_a == usize::MAX {
                break;
            }
            env.step_from_idx(best_a);
            node_id = tree[node_id].children[best_a].unwrap();
            path.push(node_id);
        }

        // 2) Expansion
        if !env.is_game_over() && !tree[node_id].untried.is_empty() {
            // Get current legal moves
            let current_legal_moves: Vec<_> = env.available_actions_ids().collect();
            
            // Keep trying untried actions until we find a legal one
            while let Some(a) = tree[node_id].untried.pop() {
                if current_legal_moves.contains(&a) {
                    env.step_from_idx(a);
                    let new_id = tree.len();
                    tree.push(Node::new(env.available_actions_ids()));
                    states.push(env.clone());
                    tree[node_id].children[a] = Some(new_id);
                    node_id = new_id;
                    path.push(node_id);
                    break;
                }
                // if action is not legal, continue popping until we find a legal one
            }
        }

        // 3) Simulation (random roll-out)
        let mut rollout = env.clone();
        while !rollout.is_game_over() {
            let legal: Vec<_> = rollout.available_actions_ids().collect();
            if legal.is_empty() { break; }
            let &a = legal.iter().choose(rng).unwrap();
            rollout.step_from_idx(a);
        }
        let reward = rollout.score();

        // 4) Back-propagation
        for &nid in &path {
            tree[nid].visits += 1;
            tree[nid].value  += reward;
        }
    }

    // ─────────────  Derive π from child visits  ─────────────────────
    let mut pi = [0.0f32; A];
    let root   = &tree[0];

    let total_visits: usize = root.children.iter()
        .filter_map(|&c| c)
        .map(|cid| tree[cid].visits)
        .sum();

    if total_visits == 0 {
        // fallback: uniform distribution avoids AllWeightsZero panic
        for p in &mut pi { *p = 1.0 / A as f32; }
    } else {
        for a in 0..A {
            if let Some(cid) = root.children[a] {
                pi[a] = tree[cid].visits as f32 / total_visits as f32;
            }
        }
    }
    pi
}

/// sample an index from a fixed‑size probability array
fn sample_from<const A: usize>(probs: [f32; A], rng: &mut impl rand::Rng) -> usize {
    let dist = WeightedIndex::new(&probs).unwrap();
    dist.sample(rng)
}

/// AlphaZero: self‑play + MCTS + joint policy/value network training
pub fn episodic_alpha_zero<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>(
    mut model: M,
    num_iterations: usize,
    episode_stop: usize,
    games_per_iteration: usize,
    learning_rate: f32,
    _weight_decay: f32,
    mcts_simulations: usize,
    c: f32,
    device: &B::Device,
    logger: &mut AlphaZeroLogger
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = AdamConfig::new()
        //.with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();

    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total_score = 0.0;
    let mut total_duration = std::time::Duration::new(0, 0);

    for ep in tqdm!(0..=num_iterations) {
        if ep % episode_stop == 0 {
            let mean = total_score / episode_stop as f32;
            let mean_duration = total_duration / episode_stop as u32;
            logger.log(ep, mean, mean_duration);
            total_score = 0.0;
        }
        
        let mut training_data = Vec::new();
        for _ in 0..games_per_iteration {
            let mut env = Env::default();

            let mut trajectory = Vec::new();
            let game_start = Instant::now();
            while !env.is_game_over() {
                let s = env.state_description();
                let pi = run_mcts_policy::<NUM_STATE_FEATURES, NUM_ACTIONS, Env, _>(
                    &env,
                    mcts_simulations,  // e.g. 100
                    c,                 // UCT exploration constant
                    &mut rng,
                );
                let a = sample_from(pi, &mut rng);
                trajectory.push((s, pi, a));
                env.step_from_idx(a);
            }
            let z = env.score().signum();
            for (s, pi, _) in trajectory {
                training_data.push((s, pi, z));
            }
            total_duration += game_start.elapsed();
            total_score += env.score();
        }

        for (state, pi, z) in training_data.drain(..) {
            let s_t = Tensor::<B, 1>::from_floats(state.as_slice(), device);
            let out = model.forward(s_t.clone());
            let (logits_p, value_v) = split_policy_value(out);

            let logp = log_softmax(logits_p);
            let pi_t = Tensor::<B, 1>::from(pi).to_device(device);
            let loss_p = -(pi_t.clone() * logp).sum();

            let z_t = Tensor::<B, 1>::from_floats([z], device);
            let loss_v = (value_v.clone() - z_t).powf_scalar(2.0);

            let loss = loss_v + loss_p;
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(learning_rate.into(), model, grads);
        }
    }

    logger.save_model(&model, num_iterations);
    println!("Mean Score : {:.3}", total_score / episode_stop as f32);
    model
}

pub fn run_alpha_zero<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>(env_name: &str,params: DeepLearningParams) {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS + 1);
    
    let name = format!("./data/alpha_zero/{}", env_name);
    let mut logger = AlphaZeroLogger::new(&name, &params);
    let trained = episodic_alpha_zero::<
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
        _,
        MyAutodiffBackend,
        Env,
    >(
        model,
        params.num_episodes,
        params.episode_stop,
        params.az_self_play_games,
        params.alpha,
        params.opt_weight_decay_penalty,
        params.mcts_simulations,
        params.az_c,
        &device,
        &mut logger
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}