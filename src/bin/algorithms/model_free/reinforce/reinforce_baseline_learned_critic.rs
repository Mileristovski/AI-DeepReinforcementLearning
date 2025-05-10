use burn::module::AutodiffModule;
use burn::optim::{Optimizer, decay::WeightDecayConfig, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{get_device, log_softmax, masked_softmax, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::distributions::Distribution;
use rand::SeedableRng;

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
    policy_lr: f32,
    critic_lr: f32,
    weight_decay: f32,
    device: &B::Device,
) -> P
where
    P::InnerModule: Forward<B = B::InnerBackend>,
    V::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut opt_pol = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();
    let mut opt_cri = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();

    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total = 0.0;

    for ep in tqdm!(0..num_episodes) {
        if ep > 0 && ep % episode_stop == 0 {
            println!("Mean Score : {:.3}", total / episode_stop as f32);
            total = 0.0;
        }

        // collect one episode
        let mut env = Env::default();
        env.set_against_random();
        env.reset();

        let mut trajectory: Vec<([f32; NUM_STATE_FEATURES], usize, f32)> = Vec::new();
        let mut s = env.state_description();

        while !env.is_game_over() {
            // policy forward
            let s_t = Tensor::<B, 1>::from_floats(s.as_slice(), device);
            let logits = policy.forward(s_t);
            let mask_t = Tensor::<B, 1>::from(env.action_mask()).to_device(device);
            let probs  = masked_softmax(logits, mask_t);            // you can integrate mask if desired
            let probs_v: Vec<f32> = probs
                .clone()
                .into_data()
                .into_vec::<f32>()
                .unwrap();
            let dist = rand::distributions::WeightedIndex::new(&probs_v).unwrap();
            let a = dist.sample(&mut rng);

            let prev = env.score();
            env.step_from_idx(a);
            let r = env.score() - prev;
            s = env.state_description();

            trajectory.push((s, a, r));
        }
        total += env.score();

        // compute returns G_t
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
            let logp = log_softmax(logits).slice([*action..action + 1]);
            let loss_pol = logp.mul_scalar(-advantage);
            let grad_pol = loss_pol.backward();
            let grads_pol = GradientsParams::from_grads(grad_pol, &policy);
            policy = opt_pol.step(policy_lr.into(), policy, grads_pol);
        }
    }

    println!("Mean Score : {:.3}", total / (episode_stop as f32));
    policy
}

pub fn run_reinforce_actor_critic<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>() {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    // policy & critic nets
    let policy = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let critic = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, 1);

    let params = DeepLearningParams::default();
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
        params.alpha,      // policy lr
        params.alpha * 2., // critic lr (for example)
        params.opt_weight_decay_penalty,
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained_policy);
}
