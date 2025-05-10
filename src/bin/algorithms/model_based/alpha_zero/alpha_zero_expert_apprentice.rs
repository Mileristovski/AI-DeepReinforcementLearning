use burn::module::AutodiffModule;
use burn::optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand::distributions::WeightedIndex;
use rand::prelude::IteratorRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use kdam::tqdm;
use rand::distributions::Distribution;

use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::services::algorithms::helpers::{get_device, log_softmax, split_policy_value, test_trained_model};
use crate::services::algorithms::model::{Forward, MyQmlp};
use std::fmt::Display;

struct Node<const A: usize> {
    visits: usize,
    value: f32,
    children: [Option<usize>; A],
    untried: Vec<usize>,
}
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
    R: Rng + ?Sized,
>(
    root_env: &Env,
    num_sim: usize,
    c: f32,
    rng: &mut R,
) -> [f32; A] {
    let mut tree: Vec<Node<A>> = Vec::new();
    let mut states: Vec<Env> = Vec::new();

    // build root
    tree.push(Node::new(root_env.available_actions_ids()));
    states.push(root_env.clone());

    for _ in 0..num_sim {
        // 1) Select
        let mut node = 0;
        let mut env = root_env.clone();
        let mut path = vec![0];

        while tree[node].untried.is_empty() && !env.is_game_over() {
            let parent_n = tree[node].visits as f32;
            let mut best_u = -f32::INFINITY;
            let mut best_a = 0;
            for a in 0..A {
                if let Some(child) = tree[node].children[a] {
                    let q = tree[child].value / tree[child].visits as f32;
                    let u = q + c * (parent_n.ln() / tree[child].visits as f32).sqrt();
                    if u > best_u {
                        best_u = u;
                        best_a = a;
                    }
                }
            }
            // descend
            env.step_from_idx(best_a);
            node = tree[node].children[best_a].unwrap();
            path.push(node);
        }

        // 2) Expand
        if !env.is_game_over() {
            let a = tree[node].untried.pop().unwrap();
            env.step_from_idx(a);
            let new_node = tree.len();
            tree.push(Node::new(env.available_actions_ids()));
            states.push(env.clone());
            tree[node].children[a] = Some(new_node);
            node = new_node;
            path.push(node);
        }

        // 3) Rollout
        let mut rollout = env.clone();
        while !rollout.is_game_over() {
            let a = rollout.available_actions_ids().choose(rng).unwrap();
            rollout.step_from_idx(a);
        }
        let reward = rollout.score();

        // 4) Back‑prop
        for &n in &path {
            tree[n].visits += 1;
            tree[n].value += reward;
        }
    }

    // compute piₑ[a] ∝ visits_of_child[a]
    let root = &tree[0];
    let mut pi = [0.0; A];
    let total: usize = root
        .children
        .iter()
        .filter_map(|&ch| ch.map(|c| tree[c].visits))
        .sum();
    for a in 0..A {
        if let Some(child) = root.children[a] {
            pi[a] = tree[child].visits as f32 / total as f32;
        }
    }
    pi
}

// —-------------------------------------------------------------------------------------
// Expert‑Apprentice AlphaZero
// —-------------------------------------------------------------------------------------
/// `apprentice_prob` ∈ [0,1] governs how often we use the **network** (apprentice) vs MCTS (expert).
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
    apprentice_prob: f32,
    c: f32,
    learning_rate: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .init();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total_score = 0.0;
    for ep in tqdm!(0..num_iterations) {
        if ep > 0 && ep % episode_stop == 0 {
            println!("Mean Score : {:.3}", total_score / episode_stop as f32);
            total_score = 0.0;
        }
        let mut training: Vec<([f32; NUM_STATE_FEATURES], [f32; NUM_ACTIONS], f32)> = Vec::new();

        for _ in 0..games_per_iteration {
            let mut env = Env::default();
            env.set_against_random();
            env.reset();

            let mut trajectory = Vec::new();
            while !env.is_game_over() {
                let pi_expert =
                    run_mcts_pi::<NUM_STATE_FEATURES, NUM_ACTIONS, Env, _>(
                        &env, mcts_sims, c, &mut rng,
                    );

                let s = env.state_description();
                let s_t = Tensor::<B, 1>::from_floats(s.as_slice(), device);
                let logits = model.forward(s_t);
                let logp = log_softmax(logits);
                let pi_net: Vec<f32> = logp
                    .clone()
                    .exp()
                    .into_data()
                    .into_vec::<f32>()
                    .unwrap();

                // 3) mix expert/apprentice
                let a = if rng.gen::<f32>() < apprentice_prob {
                    WeightedIndex::new(&pi_net).unwrap().sample(&mut rng)
                } else {
                    WeightedIndex::new(&pi_expert).unwrap().sample(&mut rng)
                };

                trajectory.push((s, pi_expert, a));
                env.step_from_idx(a);
            }

            // final z = sign(score)
            let z = env.score().signum();
            for (s, pi_e, _) in trajectory {
                training.push((s, pi_e, z));
            }

            total_score += env.score();
        }

        // now train the network on all collected (s, piₑ, z)
        for (state, pi_e, z) in training.drain(..) {
            // forward
            let s_t = Tensor::<B, 1>::from_floats(state.as_slice(), device);
            let out = model.forward(s_t.clone());
            let (logits_p, value_v) = split_policy_value::<B, NUM_ACTIONS>(out);

            let logp = log_softmax(logits_p);
            let pi_t   = Tensor::<B, 1>::from(pi_e).to_device(device);
            let loss_p = - (pi_t.clone() * logp).sum();

            let zt = Tensor::<B, 1>::from_floats([z], device);
            let loss_v = (value_v.clone() - zt).powf_scalar(2.0);

            // combined
            let loss = loss_p + loss_v;
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(learning_rate.into(), model, grads);
        }
    }
    
    println!("Mean Score : {:.3}", total_score / episode_stop as f32);
    model
}

pub fn run_alpha_zero_expert_apprentice<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>() {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model =
        MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS + 1);

    let params = DeepLearningParams::default();
    let trained = episodic_alpha_zero_expert_apprentice::<
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
        _,
        MyAutodiffBackend,
        Env,
    >(
        model,
        params.az_iterations,
        params.episode_stop,
        params.az_self_play_games,
        params.mcts_simulations,
        params.apprentice_prob,
        params.c,
        params.alpha,
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
