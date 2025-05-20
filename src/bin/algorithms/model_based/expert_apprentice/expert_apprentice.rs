use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::distributions::WeightedIndex;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use kdam::tqdm;
use rand::distributions::Distribution;

use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::services::algorithms::helpers::{get_device, log_softmax, softmax, split_policy_value, test_trained_model};
use crate::services::algorithms::model::{Forward, MyQmlp};
use std::fmt::Display;
use std::time::Instant;
use crate::services::algorithms::exports::model_based::alpha_zero::expert_apprentice::ExpertApprenticeLogger;

#[allow(dead_code)]
struct Node<const A: usize> {
    visits: usize,
    value: f32,
    children: [Option<usize>; A],
    untried: Vec<usize>,
}
#[allow(dead_code)]
impl<const A: usize> Node<A> {
    fn new(available: impl Iterator<Item=usize>) -> Self {
        Node {
            visits: 0,
            value: 0.0,
            children: [(); A].map(|_| None),
            untried: available.collect(),
        }
    }
}

fn run_mcts_pi<
    const S: usize,
    const A: usize,
    Env: DeepDiscreteActionsEnv<S, A> + Display + Default + Clone,
    R: rand::Rng + ?Sized,
>(
    root_env: &Env,
    num_sim: usize,
    c: f32,
    rng: &mut R,
) -> [f32; A] {
    // ---------- node type -------------------------------------------------
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

    // ---------- tree storage ---------------------------------------------
    let mut tree:   Vec<Node<A>> = Vec::new();
    let mut states: Vec<Env>     = Vec::new();

    tree.push(Node::new(root_env.available_actions_ids()));
    states.push(root_env.clone());

    // ---------- simulations ----------------------------------------------
    for _ in 0..num_sim {
        // 1) Selection -----------------------------------------------------
        let mut node = 0usize;
        let mut env  = root_env.clone();
        let mut path = vec![0];

        while tree[node].untried.is_empty() && !env.is_game_over() {
            let parent_n = tree[node].visits as f32;
            let mut best_a = usize::MAX;
            let mut best_u = f32::NEG_INFINITY;
            
            // Get current legal moves
            let current_legal_moves: Vec<_> = env.available_actions_ids().collect();

            for a in 0..A {
                // Only consider legal moves that have children
                if let Some(child) = tree[node].children[a] {
                    if current_legal_moves.contains(&a) {
                        let q = tree[child].value / tree[child].visits as f32;
                        let u = q + c * (parent_n.ln() / tree[child].visits as f32).sqrt();
                        if u > best_u {
                            best_u = u;
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
            node = tree[node].children[best_a].unwrap();
            path.push(node);
        }

        // 2) Expansion -----------------------------------------------------
        if !env.is_game_over() && !tree[node].untried.is_empty() {
            let current_legal_moves: Vec<_> = env.available_actions_ids().collect();
            
            // Keep trying untried actions until we find a legal one
            while let Some(a) = tree[node].untried.pop() {
                if current_legal_moves.contains(&a) {
                    env.step_from_idx(a);
                    let new_id = tree.len();
                    tree.push(Node::new(env.available_actions_ids()));
                    states.push(env.clone());
                    tree[node].children[a] = Some(new_id);
                    node = new_id;
                    path.push(node);
                    break;
                }
                // if action is not legal, continue popping until we find a legal one
            }
        }

        // 3) Roll-out (random policy) -------------------------------------
        let mut rollout = env.clone();
        while !rollout.is_game_over() {
            if let Some(a) = rollout.available_actions_ids().choose(rng) {
                rollout.step_from_idx(a);
            } else {
                break; // no legal actions – shouldn’t happen
            }
        }
        let reward = rollout.score();

        // 4) Back-propagation ---------------------------------------------
        for &nid in &path {
            tree[nid].visits += 1;
            tree[nid].value  += reward;
        }
    }

    // ---------- visits → π -----------------------------------------------
    let mut pi = [0.0f32; A];
    let root = &tree[0];
    let total: usize = root.children
        .iter()
        .filter_map(|&c| c.map(|id| tree[id].visits))
        .sum();

    if total == 0 {
        // fallback uniform
        for p in &mut pi { *p = 1.0 / A as f32 }
    } else {
        for a in 0..A {
            if let Some(cid) = root.children[a] {
                pi[a] = tree[cid].visits as f32 / total as f32;
            }
        }
    }
    pi
}

pub fn episodic_alpha_zero_expert_apprentice<
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
    mcts_sims: usize,
    c: f32,
    learning_rate: f32,
    _weight_decay: f32,
    device: &B::Device,
    logger: &mut ExpertApprenticeLogger
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = AdamConfig::new()
        // .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total_score = 0.0;
    let mut total_duration = std::time::Duration::new(0, 0);
    for ep in tqdm!(0..num_iterations) {
        if ep % episode_stop == 0 {
            let mean = total_score / episode_stop as f32;
            let mean_duration = total_duration / episode_stop as u32;
            logger.log(ep, mean, mean_duration);
            total_score = 0.0;
        }

        let mut training: Vec<([f32; NUM_STATE_FEATURES], [f32; NUM_ACTIONS], f32)> = Vec::new();

        for _ in 0..games_per_iteration {
            let mut env = Env::default();
            let mut trajectory = Vec::new();
            
            let game_start = Instant::now();
            while !env.is_game_over() {
                let pi_expert =
                    run_mcts_pi::<NUM_STATE_FEATURES, NUM_ACTIONS, Env, _>(
                        &env, mcts_sims, c, &mut rng,
                    );

                let s = env.state_description();
                let s_t = Tensor::<B, 1>::from_floats(s.as_slice(), device);
                let out = model.forward(s_t);

                let (policy_logits, _value) =
                    split_policy_value::<B, NUM_ACTIONS>(out);

                let mask_t = Tensor::<B,1>::from_floats(env.action_mask(), device);
                let masked = policy_logits + (mask_t.clone() - 1.0) * 1e9;
                let probs  = softmax(masked);

                let dist   = WeightedIndex::new(
                    &probs.clone().into_data().into_vec::<f32>().unwrap()
                ).unwrap();
                let a      = dist.sample(&mut rng);


                trajectory.push((s, pi_expert, a));
                env.step_from_idx(a);
            }

            let z = env.score().signum();
            for (s, pi_e, _) in trajectory {
                training.push((s, pi_e, z));
            }
            total_duration += game_start.elapsed();
            total_score += env.score();
        }

        for (state, pi_e, z) in training.drain(..) {
            let s_t = Tensor::<B, 1>::from_floats(state.as_slice(), device);
            let out = model.forward(s_t.clone());
            let (logits_p, value_v) = split_policy_value::<B, NUM_ACTIONS>(out);

            let logp = log_softmax(logits_p);
            let pi_t   = Tensor::<B, 1>::from(pi_e).to_device(device);
            let loss_p = - (pi_t.clone() * logp).sum();

            let zt = Tensor::<B, 1>::from_floats([z], device);
            let loss_v = (value_v.clone() - zt).powf_scalar(2.0);

            let loss = loss_p + loss_v;
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(learning_rate.into(), model, grads);
        }
    }
    
    logger.save_model(&model, num_iterations);
    println!("Mean Score : {:.3}", total_score / episode_stop as f32);
    model
}

pub fn expert_apprentice<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>(env_name: &str,params: &DeepLearningParams) {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model =
        MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS + 1);

    let name = format!("./data/{}/expert_apprentise", env_name);
    let mut logger = ExpertApprenticeLogger::new(&name, env_name.parse().unwrap(), &params);
    let trained = episodic_alpha_zero_expert_apprentice::<
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
        params.mcts_simulations,
        params.az_c,
        params.alpha,
        params.opt_weight_decay_penalty,
        &device,
        &mut logger,
    );

    if params.run_test {
        test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
    }
}