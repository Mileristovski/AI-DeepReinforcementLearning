use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::fmt::Display;
use std::time::Instant;
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::services::algorithms::exports::model_free::reinforce::reinforce_mean_baseline::ReinforceBaselineLogger;
use crate::services::algorithms::helpers::{
    get_device, masked_log_softmax, masked_softmax, test_trained_model,
};
use crate::services::algorithms::model::{Forward, MyQmlp};

// ────────────────────────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
pub fn episodic_reinforce_baseline<
    const N_S: usize,
    const N_A: usize,
    M,
    B,
    Env,
>(
    mut model: M,
    num_episodes: usize,
    episode_stop: usize,
    gamma: f32,
    lr: f32,
    device: &B::Device,
    logger: &mut ReinforceBaselineLogger
) -> M
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    M::InnerModule: Forward<B = B::InnerBackend>,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Display + Default,
{
    let mut opt = AdamConfig::new()
        // .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();
    let mut rng   = Xoshiro256PlusPlus::from_entropy();
    let mut score_sum = 0.0f32;
    let mut total_duration = std::time::Duration::new(0, 0);

    // each episode ---------------------------------------------------------
    for ep in tqdm!(0..num_episodes) {
        if ep % episode_stop == 0 {
            let mean = score_sum / episode_stop as f32;
            let mean_duration = total_duration / episode_stop as u32;
            logger.log(ep, mean, mean_duration);
            score_sum = 0.0;
        }

        let mut env = Env::default();

        // (state, mask, action, reward) per time‑step ----------------------
        let mut traj: Vec<([f32; N_S], [f32; N_A], usize, f32)> = Vec::new();

        let game_start = Instant::now();
        while !env.is_game_over() {
            let s = env.state_description();
            let mask_arr = env.action_mask();

            let logits   = model.forward(Tensor::<B, 1>::from_floats(s.as_slice(), device));
            let mask_t   = Tensor::<B, 1>::from(mask_arr).to_device(device);
            let probs    = masked_softmax(logits, mask_t);
            let weights  = probs.into_data().into_vec::<f32>().unwrap();
            let dist     = WeightedIndex::new(&weights).expect("invalid probs");
            let a        = dist.sample(&mut rng);

            let prev_score = env.score();
            env.step_from_idx(a);
            let r = env.score() - prev_score;

            traj.push((s, mask_arr, a, r));
        }
        total_duration += game_start.elapsed();
        score_sum += env.score();

        // Monte‑Carlo returns ---------------------------------------------
        let mut returns = Vec::with_capacity(traj.len());
        let mut g = 0.0f32;
        for &(_, _, _, r) in traj.iter().rev() {
            g = r + gamma * g;
            returns.push(g);
        }
        returns.reverse();

        let baseline = returns.iter().sum::<f32>() / returns.len() as f32;

        // one gradient step per time‑step (could be batched) --------------
        for ((state, mask_arr, action, _), &g_t) in traj.iter().zip(returns.iter()) {
            let advantage = g_t - baseline;
            if advantage.abs() < 1e-6 { continue; } // tiny grads – skip

            let logits = model.forward(Tensor::<B, 1>::from_floats(state.as_slice(), device));
            let mask_t = Tensor::<B, 1>::from(*mask_arr).to_device(device);
            let log_probs = masked_log_softmax(logits, mask_t);

            let logp = log_probs.clone().slice([*action .. (*action + 1)]);
            let loss = logp.mul_scalar(-advantage); // minimise –advantage * log π

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = opt.step(lr.into(), model, grads);
        }
    }
    logger.save_model(&model, num_episodes);
    println!("Mean Score : {:.3}", score_sum / episode_stop as f32);
    model
}

// ────────────────────────────────────────────────────────────────────────────
pub fn run_reinforce_baseline<
    const N_S: usize,
    const N_A: usize,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Display,
>(env_name: &str,p: &DeepLearningParams) {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, N_S, N_A);
    
    let name = format!("./data/{}/reinforce_mb", env_name);
    let mut logger = ReinforceBaselineLogger::new(&name, &p);
    let trained = episodic_reinforce_baseline::<N_S, N_A, _, MyAutodiffBackend, Env>(
        model,
        p.num_episodes,
        p.episode_stop,
        p.gamma,
        p.alpha,  // learning‑rate
        &device,
        &mut logger
    );

    if p.run_test {
        test_trained_model::<N_S, N_A, Env>(&device, trained);
    }
}
