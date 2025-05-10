use burn::module::AutodiffModule;
use burn::optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{softmax, log_softmax, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;

pub fn episodic_ppo_a2c<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    P: Forward<B = B> + AutodiffModule<B> + Clone,  // policy network
    V: Forward<B = B> + AutodiffModule<B> + Clone,  // value network
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default,
>(
    mut policy: P,
    mut critic: V,
    num_episodes: usize,
    episode_stop: usize,
    gamma: f32,
    entropy_coef: f32,
    policy_lr: f32,
    critic_lr: f32,
    device: &B::Device,
) -> P
where
    P::InnerModule: Forward<B = B::InnerBackend>,
    V::InnerModule: Forward<B = B::InnerBackend>,
{
    // two separate optimizers
    let mut opt_pol = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();
    let mut opt_cri = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();

    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total = 0.0;

    for ep in tqdm!(0..num_episodes) {
        // optionally log
        if ep > 0 && ep % episode_stop == 0 {
            println!("Mean Score : {:.3}", total / episode_stop as f32);
            total = 0.0;
        }

        // run one episode, collecting up to n_step transitions at a time
        let mut env = Env::default();
        env.set_against_random();
        env.reset();

        // store (state, action, reward)
        let mut trajectory: Vec<([f32; NUM_STATE_FEATURES], usize, f32)> = Vec::new();
        let mut s = env.state_description();

        while !env.is_game_over() {
            // policy → distribution
            let s_t = Tensor::<B, 1>::from_floats(s.as_slice(), device);
            let logits = policy.forward(s_t.clone());
            let probs = softmax(logits);
            let p_vec: Vec<f32> = probs.clone().into_data().into_vec().unwrap();
            let dist = WeightedIndex::new(&p_vec).unwrap();
            let a = dist.sample(&mut rng);

            let prev = env.score();
            env.step_from_idx(a);
            let r = env.score() - prev;
            let s2 = env.state_description();

            trajectory.push((s, a, r));
            s = s2;

            if trajectory.len() >= episode_stop || env.is_game_over() {
                let mut r = if env.is_game_over() {
                    0.0
                } else {
                    let s_tn = Tensor::<B, 1>::from_floats(s.as_slice(), device);
                    critic.forward(s_tn).slice([0..1]).into_scalar()
                };

                for &(state, action, reward) in trajectory.iter().rev() {
                    r = reward + gamma * r;

                    let s_tc = Tensor::<B, 1>::from_floats(state.as_slice(), device);
                    let v_s = critic.forward(s_tc.clone()).slice([0..1]);
                    let loss_cri = (v_s - Tensor::from([r]).to_device(device)).powf_scalar(2.0);
                    let grad_cri = loss_cri.backward();
                    let grads_cri = GradientsParams::from_grads(grad_cri, &critic);
                    critic = opt_cri.step(critic_lr.into(), critic, grads_cri);

                    let baseline = critic.forward(s_tc.clone()).slice([0..1]).detach().into_scalar();
                    let advantage = r - baseline;
                    let logits_p = policy.forward(s_tc);
                    let logp = log_softmax(logits_p.clone()).slice([action..action+1]);
                    let entropy = - (softmax(logits_p.clone()) * log_softmax(logits_p)).sum();
                    let loss_pol = logp.mul_scalar(-advantage) - entropy.mul_scalar(entropy_coef);
                    let grad_pol = loss_pol.backward();
                    let grads_pol = GradientsParams::from_grads(grad_pol, &policy);
                    policy = opt_pol.step(policy_lr.into(), policy, grads_pol);
                }

                trajectory.clear();
            }
        }

        total += env.score();
    }

    println!(
        "Final mean score (last {} eps): {:.3}",
        episode_stop,
        total / episode_stop as f32
    );
    policy
}

pub fn run_ppo_a2c<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>() {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let policy = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let critic = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, 1);

    // hyperparameters
    let params = DeepLearningParams::default();
    let trained = episodic_ppo_a2c::<
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
        params.n_step,
        params.gamma,
        params.entropy_coef,
        params.policy_lr,
        params.critic_lr,
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
