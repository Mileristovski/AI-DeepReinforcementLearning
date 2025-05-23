use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{get_device, log_softmax, masked_softmax, softmax, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use std::time::Instant;
use kdam::tqdm;
use rand::distributions::Distribution;
use rand::SeedableRng;
use crate::services::algorithms::exports::model_free::reinforce::reinforce_baseline_ac::ReinforceLCLogger;

pub fn episodic_actor_critic<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    P: Forward<B = B> + AutodiffModule<B> + Clone,  // policy net
    V: Forward<B = B> + AutodiffModule<B> + Clone,  // value  net
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default,
>(
    mut policy: P,
    mut critic: V,
    num_episodes: usize,
    episode_stop: usize,
    gamma: f32,
    ent_coef: f32,
    policy_lr: f32,
    critic_lr: f32,
    _weight_decay: f32,
    device: &B::Device,
    logger: &mut ReinforceLCLogger
) -> P
where
    P::InnerModule: Forward<B = B::InnerBackend>,
    V::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut opt_pol = AdamConfig::new()
        // .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();
    let mut opt_cri = AdamConfig::new()
        // .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();

    let mut env = Env::default();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total = 0.0;
    let mut total_duration = std::time::Duration::new(0, 0);

    for ep in tqdm!(0..num_episodes) {
        if ep % episode_stop == 0 {
            let mean = total / episode_stop as f32;
            let mean_duration = total_duration / episode_stop as u32;
            logger.log(ep, mean, mean_duration);
            total = 0.0;
        }

        // collect one episode
        let mut trajectory: Vec<([f32; NUM_STATE_FEATURES], usize, f32)> = Vec::new();
        let mut s = env.state_description();

        let game_start = Instant::now();
        while !env.is_game_over() {
            // policy forward
            let s_t = Tensor::<B, 1>::from_floats(s.as_slice(), device);
            let logits = policy.forward(s_t);

            let mask   = env.action_mask();
            let mask_t = Tensor::<B,1>::from(mask).to_device(device);

            // Apply softmask with mask
            let dist = masked_softmax::<B, NUM_ACTIONS>(logits, mask_t, device);
            let a = dist.sample(&mut rng);

            let prev = env.score();
            env.step_from_idx(a);

            let r = env.score() - prev;
            let s2   = env.state_description();

            trajectory.push((s, a, r));
            s   = s2;
        }
        total_duration += game_start.elapsed();
        total += env.score();

        // compute returns g_t
        let mut returns = Vec::with_capacity(trajectory.len());
        let mut g = 0.0;
        for &(_, _, r) in trajectory.iter().rev() {
            g = r + gamma * g;
            returns.push(g);
        }
        returns.reverse();

        // update both networks
        for ((state, action, _), &g_t) in trajectory.iter().zip(returns.iter()) {
            let s_t = Tensor::<B, 1>::from_floats(state.as_slice(), device);
            let v_s = critic.forward(s_t.clone()).slice([0..1]);
            let loss_cri = (v_s - Tensor::from([g_t]).to_device(device)).powf_scalar(2.0);
            let grad_cri = loss_cri.backward();
            let grads_cri = GradientsParams::from_grads(grad_cri, &critic);
            critic = opt_cri.step(critic_lr.into(), critic, grads_cri);

            let baseline = critic.forward(s_t.clone()).slice([0..1]).detach().into_scalar();
            let advantage = g_t - baseline;
            let logits = policy.forward(s_t);

            let log_probs = log_softmax(logits.clone());
            let logp = log_probs.clone().slice([*action .. action + 1]);
            let entropy = - (softmax(logits.clone()) * log_probs).sum();

            let loss_pol = logp.mul_scalar(-advantage) - entropy.mul_scalar(ent_coef);
            let grad_pol = loss_pol.backward();
            let grads_pol = GradientsParams::from_grads(grad_pol, &policy);
            policy = opt_pol.step(policy_lr.into(), policy, grads_pol);
        }
        env.reset();
    }

    logger.save_model(&policy, num_episodes);
    println!("Mean Score : {:.3}", total / (episode_stop as f32));
    policy
}

pub fn run_reinforce_actor_critic<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(env_name: &str,params: &DeepLearningParams) {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    // policy & critic nets
    let policy = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let critic = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, 1);
    
    let name = format!("./data/{}/reinforce_lc", env_name);
    let mut logger = ReinforceLCLogger::new(&name, env_name.parse().unwrap(), &params);
    let trained_policy = episodic_actor_critic::<
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
        _,
        _,
        MyAutodiffBackend,
        Env,
    >(
        policy,
        critic,
        params.num_episodes,
        params.episode_stop,
        params.gamma,
        params.ac_entropy_coef,
        params.alpha,      // policy lr
        params.alpha * 2., // critic lr (for example)
        params.opt_weight_decay_penalty,
        &device,
        &mut logger,
    );

    if params.run_test {
        test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained_policy);
    }
}
