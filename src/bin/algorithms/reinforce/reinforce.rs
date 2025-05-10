use burn::module::AutodiffModule;
use burn::optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algo_helper::helpers::{get_device, log_softmax, softmax, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algo_helper::qmlp::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;

/// Vanilla REINFORCE (no baseline) with episodic returns, masked so illegal actions never sampled.
pub fn episodic_reinforce<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default,
>(
    mut model: M,
    num_episodes: usize,
    episode_stop: usize,
    gamma: f32,
    learning_rate: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();

    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total_score = 0.0;
    // large negative for illegal actions
    let neg_inf = Tensor::<B, 1>::from_floats([-1e9; NUM_ACTIONS], device);

    for ep in tqdm!(0..num_episodes) {
        if ep > 0 && ep % episode_stop == 0 {
            println!("Ep {:>4}/{}  mean score {:.3}", ep, num_episodes, total_score / episode_stop as f32);
            total_score = 0.0;
        }

        let mut env = Env::default();
        env.set_against_random();
        env.reset();

        let mut trajectory: Vec<([f32; NUM_STATE_FEATURES], usize, f32)> = Vec::new();
        let mut s = env.state_description();

        while !env.is_game_over() {
            // 1) compute masked action probabilities
            let s_t = Tensor::<B, 1>::from_floats(s.as_slice(), device);
            let logits = model.forward(s_t);

            // mask out illegal actions
            let mask_arr = env.action_mask();
            let mask_t = Tensor::<B, 1>::from(mask_arr).to_device(device);
            let adjusted = logits.clone() * mask_t.clone()
                + neg_inf.clone() * (mask_t.clone().neg().add_scalar(1.0));

            let probs = softmax(adjusted);
            let probs_vec = probs.clone().into_data().into_vec::<f32>().unwrap();
            let dist = WeightedIndex::new(&probs_vec).unwrap();
            let a = dist.sample(&mut rng);

            // 2) step environment
            let prev_score = env.score();
            env.step_from_idx(a);
            let reward = env.score() - prev_score;

            // record before updating state
            trajectory.push((s, a, reward));
            s = env.state_description();
        }

        // compute returns G_t
        let mut returns = Vec::with_capacity(trajectory.len());
        let mut G = 0.0;
        for &(_, _, r) in trajectory.iter().rev() {
            G = r + gamma * G;
            returns.push(G);
        }
        returns.reverse();

        // policy update
        for ((state, action, _), &G_t) in trajectory.iter().zip(returns.iter()) {
            let s_t = Tensor::<B, 1>::from_floats(state.as_slice(), device);
            let logits = model.forward(s_t);

            // re-mask
            let mask_arr = env.action_mask();
            let mask_t = Tensor::<B, 1>::from(mask_arr).to_device(device);
            let adjusted = logits.clone() * mask_t.clone()
                + neg_inf.clone() * (mask_t.clone().neg().add_scalar(1.0));

            let log_probs = log_softmax(adjusted);
            let logp = log_probs.clone().slice([*action..*action + 1]);

            let loss = logp.mul_scalar(-G_t);
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(learning_rate.into(), model, grads);
        }

        total_score += env.score();
    }
    println!("Mean Score : {:.3}", total_score / (episode_stop as f32));
    model
}

/// Helper to run REINFORCE end‑to‑end.
pub fn run_reinforce<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>() {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let params = DeepLearningParams::default();

    let trained = episodic_reinforce::<
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
        _,
        MyAutodiffBackend,
        Env,
    >(
        model,
        params.num_episodes,
        params.episode_stop,
        params.gamma,
        params.alpha, // use alpha as policy‐learning‐rate
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
