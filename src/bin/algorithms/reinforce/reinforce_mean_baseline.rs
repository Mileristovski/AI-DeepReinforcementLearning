use burn::module::AutodiffModule;
use burn::optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algo_helper::helpers::{get_device, log_softmax, masked_log_softmax, masked_softmax, softmax, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algo_helper::qmlp::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::distributions::Distribution;
use rand::SeedableRng;

pub fn episodic_reinforce_baseline<
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

    for ep in tqdm!(0..num_episodes) {
        if ep > 0 && ep % episode_stop == 0 {
            println!(
                "Ep {:>4}/{}  mean score {:.3}",
                ep,
                num_episodes,
                total_score / episode_stop as f32
            );
            total_score = 0.0;
        }

        // collect one episode
        let mut env = Env::default();
        env.set_against_random();
        env.reset();

        let mut trajectory: Vec<([f32; NUM_STATE_FEATURES], usize, f32)> = Vec::new();
        let mut s = env.state_description();

        while !env.is_game_over() {
            // 1) forward policy
            let s_t = Tensor::<B, 1>::from_floats(s.as_slice(), device);
            let logits = model.forward(s_t);

            // 2) mask illegal actions by converting the mask-array into a Tensor
            let mask_arr = env.action_mask();
            let mask_t = Tensor::<B, 1>::from(mask_arr).to_device(device);

            // 3) get action probabilities
            let masked_probs = masked_softmax(logits.clone(), mask_t.clone());

            // 4) sample
            let probs_vec = masked_probs
                .clone()
                .into_data()
                .into_vec::<f32>()
                .unwrap();
            let dist = rand::distributions::WeightedIndex::new(&probs_vec).unwrap();
            let a = dist.sample(&mut rng);

            // 5) step env
            let prev_score = env.score();
            env.step_from_idx(a);
            let reward = env.score() - prev_score;
            s = env.state_description();

            trajectory.push((s, a, reward));
        }

        total_score += env.score();

        // compute returns Gₜ
        let mut returns = Vec::with_capacity(trajectory.len());
        let mut G = 0.0;
        for &(_, _, r) in trajectory.iter().rev() {
            G = r + gamma * G;
            returns.push(G);
        }
        returns.reverse();

        // baseline = mean of episode returns
        let baseline = returns.iter().sum::<f32>() / returns.len() as f32;

        // policy updates
        for ((state, action, _), &G_t) in trajectory.iter().zip(returns.iter()) {
            let advantage = G_t - baseline;

            let s_t = Tensor::<B, 1>::from_floats(state.as_slice(), device);
            let logits = model.forward(s_t.clone());

            // again mask with a tensor
            let mask_arr = env.action_mask();
            let mask_t = Tensor::<B, 1>::from(mask_arr).to_device(device);

            // compute log‐prob of chosen action
            let log_probs = masked_log_softmax(logits, mask_t);
            let logp = log_probs.slice([*action..action + 1]);

            // gradient ascent on E[log π * (Gₜ – baseline)]
            let loss = logp.mul_scalar(-advantage);
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(learning_rate.into(), model, grads);
        }
    }

    println!(
        "Final mean score (last {} eps): {:.3}",
        episode_stop,
        total_score / episode_stop as f32
    );
    model
}

pub fn run_reinforce_baseline<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>() {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    // policy network
    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);

    // hyperparams
    let params = DeepLearningParams::default();
    let trained = episodic_reinforce_baseline::<
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
        params.alpha,    // use alpha as policy LR
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
