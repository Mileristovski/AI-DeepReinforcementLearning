use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{epsilon_greedy_action, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use std::time::Instant;
use kdam::tqdm;
use rand::SeedableRng;
use crate::services::algorithms::exports::model_free::q_learning::deep_q_learning::DqnLogger;

/// Vanilla 1-step Deep-Q-Learning (no replay / no DDQN).
fn episodic_deep_q_learning<
    const N_S: usize,
    const N_A: usize,
    M,
    B,
    Env,
>(
    mut online: M,
    num_episodes: usize,
    episode_stop : usize,
    gamma : f32,
    lr    : f32,
    eps_start: f32,
    eps_final: f32,
    minus_one: &Tensor<B,1>,
    plus_one : &Tensor<B,1>,
    fmin_vec : &Tensor<B,1>,
    _wd: f32,
    device: &B::Device,
    logger: &mut DqnLogger,
) -> M
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    M::InnerModule: Forward<B = B::InnerBackend>,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Display + Default,
{
    const TARGET_SYNC_EVERY: usize = 1_000;           // steps
    let mut step_cnt  = 0usize;
    let mut target    = online.clone();               // frozen target
    let mut opt       = AdamConfig::new().init();
    let mut rng       = Xoshiro256PlusPlus::from_entropy();
    let mut score_sum = 0.0;
    let mut total_duration = std::time::Duration::new(0, 0);

    for ep in tqdm!(0..num_episodes) {
        if ep > 0 && ep % episode_stop == 0 {
            let mean = score_sum / episode_stop as f32;
            let mean_duration = total_duration / episode_stop as u32;
            logger.log(ep, mean, mean_duration);
            score_sum = 0.0;
        }
        let frac = ep as f32 / num_episodes as f32;
        let eps  = (1. - frac) * eps_start + frac * eps_final;

        let mut env = Env::default();
        let mut s = env.state_description();

        let game_start = Instant::now();
        while !env.is_game_over() {
            step_cnt += 1;
            
            let q_s = online.forward(Tensor::<B,1>::from_floats(s.as_slice(), device));
            let a   = epsilon_greedy_action::<B, N_S, N_A>(
                &q_s,
                &Tensor::from(env.action_mask()).to_device(device),
                minus_one, plus_one, fmin_vec,
                env.available_actions_ids(),
                eps, &mut rng);

            let prev = env.score(); env.step_from_idx(a);
            let r    = env.score() - prev;
            let done = env.is_game_over();
            let s2   = env.state_description();

            let y = if done {
                Tensor::<B,1>::from([r]).to_device(device)
            } else {
                let q_next = target
                    .forward(Tensor::<B,1>::from_floats(s2.as_slice(), device))
                    .detach();                         // ← no grad
                let max_q  = q_next.max().into_scalar();
                Tensor::<B,1>::from([r + gamma * max_q]).to_device(device)
            };

            let q_sa = q_s.slice([a..a+1]);
            let loss  = (q_sa - y).powf_scalar(2.0);
            let grads = GradientsParams::from_grads(loss.backward(), &online);
            online = opt.step(lr.into(), online, grads);
            
            if step_cnt % TARGET_SYNC_EVERY == 0 {
                target = online.clone();
            }

            s = s2;
        }
        total_duration += game_start.elapsed();
        score_sum += env.score();
    }
    
    logger.save_model(&online, num_episodes);
    println!("Mean Score : {:.3}", score_sum / episode_stop as f32);
    online
}


pub fn run_deep_q_learning<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(env_name: &str,params: DeepLearningParams)
{
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let minus_one = Tensor::from_floats([-1.0; NUM_ACTIONS], &device);
    let plus_one  = Tensor::from_floats([ 1.0; NUM_ACTIONS], &device);
    let fmin_vec  = Tensor::from_floats([f32::MIN; NUM_ACTIONS], &device);

    let name = format!("./data/dql/{}", env_name);
    let mut logger = DqnLogger::new(&name, &params);
    let trained = episodic_deep_q_learning::<
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
        _,
        MyAutodiffBackend,
        Env
    >(
        model,
        params.num_episodes,
        params.episode_stop,
        params.gamma,
        params.alpha,
        params.start_epsilon,
        params.final_epsilon,
        &minus_one,
        &plus_one,
        &fmin_vec,
        params.opt_weight_decay_penalty,
        &device,
        &mut logger,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
