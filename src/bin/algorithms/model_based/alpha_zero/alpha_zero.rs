// --- alpha_zero.rs ---

use burn::module::AutodiffModule;
use burn::optim::{Optimizer, decay::WeightDecayConfig, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{get_device, log_softmax, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::SeedableRng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

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
fn run_mcts_policy<
    const S: usize,
    const A: usize,
    Env: DeepDiscreteActionsEnv<S, A> + Clone
>(
    root_env: &Env,
    // num_sims: usize
) -> [f32; A] {
    #[derive(Clone)]
    struct Node<const A: usize> { visits: usize, children: [Option<usize>; A] }

    let mut tree: Vec<Node<A>> = Vec::new();
    let mut states: Vec<Env> = Vec::new();
    
    let root_id = 0;
    let root_node = Node { visits: 0, children: [(); A].map(|_| None) };
    tree.push(root_node.clone());
    states.push(root_env.clone());

    /*for _ in 0..num_sims {
        let node = root_id;
    }*/

    let mut visits = [0f32; A];
    let root = &tree[root_id];
    for (a, &opt_child) in root.children.iter().enumerate() {
        if let Some(child) = opt_child {
            visits[a] = tree[child].visits as f32;
        }
    }
    
    let sum: f32 = visits.iter().sum();
    if sum > 0.0 {
        for v in &mut visits {
            *v /= sum;
        }
    }
    visits
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
    // mcts_sims: usize,
    learning_rate: f32,
    weight_decay: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();

    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total_score = 0.0;

    for ep in tqdm!(0..num_iterations) {
        if ep > 0 && ep % episode_stop == 0 {
            println!("Mean Score : {:.3}", total_score / episode_stop as f32);
            total_score = 0.0;
        }
        let mut training_data = Vec::new();
        for _ in 0..games_per_iteration {
            let mut env = Env::default();
            env.set_against_random();
            env.reset();

            let mut trajectory = Vec::new();
            while !env.is_game_over() {
                let s = env.state_description();
                let pi = run_mcts_policy::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(
                    &env,
                    // mcts_sims
                );
                let a = sample_from(pi, &mut rng);
                trajectory.push((s, pi, a));
                env.step_from_idx(a);
            }
            let z = env.score().signum();
            for (s, pi, _) in trajectory {
                training_data.push((s, pi, z));
            }

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
    
    println!("Mean Score : {:.3}", total_score / episode_stop as f32);
    model
}

pub fn run_alpha_zero<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>() {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS + 1);
    let params = DeepLearningParams::default();
    let trained = episodic_alpha_zero::<
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
        // params.mcts_simulations,
        params.alpha,
        params.opt_weight_decay_penalty,
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
