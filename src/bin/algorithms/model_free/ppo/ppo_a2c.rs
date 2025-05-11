use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{softmax, log_softmax, get_device, test_trained_model, masked_softmax};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use std::time::Instant;
use kdam::tqdm;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;
use crate::services::algorithms::exports::model_free::a2c::A2cLogger;

fn episodic_a2c<
    const N_S: usize,
    const N_A: usize,
    P, V, B, Env,
>(
    mut policy : P,
    mut critic : V,
    num_episodes : usize,
    log_every    : usize,
    n_step       : usize,
    gamma        : f32,
    ent_coef     : f32,
    lr_pol       : f32,
    lr_val       : f32,
    _wd          : f32,
    device       : &B::Device,
    logger: &mut A2cLogger
) -> P
where
    B : AutodiffBackend<FloatElem = f32, IntElem = i64>,
    P : Forward<B = B> + AutodiffModule<B> + Clone,
    V : Forward<B = B> + AutodiffModule<B> + Clone,
    P::InnerModule : Forward<B = B::InnerBackend>,
    V::InnerModule : Forward<B = B::InnerBackend>,
    Env : DeepDiscreteActionsEnv<N_S, N_A> + Display + Default,
{
    let mut opt_pol = AdamConfig::new().init();
    let mut opt_val = AdamConfig::new().init();
    let mut rng     = Xoshiro256PlusPlus::from_entropy();

    let mut score_sum = 0.0;
    let mut total_duration = std::time::Duration::new(0, 0);
    let mut env       = Env::default(); 

    for ep in tqdm!(0..num_episodes) {
        if ep % log_every == 0 {
            let mean = score_sum / log_every as f32;
            let mean_duration = total_duration / log_every as u32;
            logger.log(ep, mean, mean_duration);
            score_sum = 0.0;
        }

        let mut traj: Vec<([f32; N_S], usize, f32)> = Vec::new();
        let mut s = env.state_description();

        let game_start = Instant::now();
        while !env.is_game_over() {
            let logits = policy.forward(Tensor::<B,1>::from_floats(s.as_slice(), device));
            let mask_array = env.action_mask();  // [48] of 0.0 or 1.0
            let mask_t     = Tensor::<B,1>::from_floats(mask_array, &device);

            // 3. masked softmax: illegal → p = 0
            let probs      = masked_softmax(logits.clone(), mask_t.clone());
            let probs_vec  = probs.clone().into_data().into_vec::<f32>().unwrap();
            let dist       = WeightedIndex::new(&probs_vec).unwrap();
            let a          = dist.sample(&mut rng);

            let prev = env.score(); env.step_from_idx(a);
            let r    = env.score() - prev;
            let s2   = env.state_description();

            traj.push((s, a, r));
            s = s2;

            if traj.len() >= n_step || env.is_game_over() {
                // bootstrap value for last state
                let mut r = if env.is_game_over() {
                    0.0
                } else {
                    critic.forward(
                        Tensor::<B,1>::from_floats(s.as_slice(), device)
                    ).slice([0..1]).into_scalar()
                };

                for (state, action, reward) in traj.iter().rev() {
                    r = reward + gamma * r;

                    let s_t = Tensor::<B,1>::from_floats(state.as_slice(), device);

                    let v_pred = critic.forward(s_t.clone()).slice([0..1]);
                    let loss_v = (v_pred.clone() - Tensor::from([r]).to_device(device))
                        .powf_scalar(2.0);
                    let grad_v = loss_v.backward();
                    let grads_v = GradientsParams::from_grads(grad_v, &critic);
                    critic = opt_val.step(lr_val.into(), critic, grads_v);

                    let baseline = v_pred.detach().into_scalar(); // OLD value
                    let advantage = r - baseline;

                    let logits_p = policy.forward(s_t.clone());
                    let log_probs = log_softmax(logits_p.clone());

                    let logp = log_probs.clone().slice([*action .. action + 1]);
                    let entropy = - (softmax(logits_p.clone()) * log_probs).sum();

                    let loss_p = logp.mul_scalar(-advantage) - entropy.mul_scalar(ent_coef);
                    let grad_p = loss_p.backward();
                    let grads_p = GradientsParams::from_grads(grad_p, &policy);
                    policy = opt_pol.step(lr_pol.into(), policy, grads_p);
                }
                traj.clear();
            }
        }
        total_duration += game_start.elapsed();
        score_sum += env.score();
        env.reset();
    }

    logger.save_model(&policy, num_episodes);
    println!("Mean Score : {:.3}", score_sum / (log_every+1) as f32);
    policy
}

pub fn run_ppo_a2c<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(env_name: &str,params: DeepLearningParams) {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let policy = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let critic = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, 1);
    // hyperparameters
    
    let name = format!("./data/a2c/{}", env_name);
    let mut logger = A2cLogger::new(&name, &params);
    let trained = episodic_a2c::<
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
        params.ac_n_step,
        params.gamma,
        params.ac_entropy_coef,
        params.ac_policy_lr,
        params.ac_critic_lr,
        params.opt_weight_decay_penalty,
        &device,
        &mut logger
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
