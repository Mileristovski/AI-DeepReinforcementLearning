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
use crate::services::algorithms::exports::model_free::q_learning::double_deep_q_learning::DoubleDqnLogger;

/// Double Deep Q‑Learning without experience replay.
fn episodic_double_deep_q_learning<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M,
    B,
    Env,
>(
    mut model: M,
    num_episodes: usize,
    episode_stop: usize,
    gamma: f32,
    alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    minus_one: &Tensor<B, 1>,
    plus_one:  &Tensor<B, 1>,
    fmin_vec:  &Tensor<B, 1>,
    _weight_decay: f32,
    device: &B::Device,
    logger: &mut DoubleDqnLogger
) -> M
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    M::InnerModule: Forward<B = B::InnerBackend>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default,
{
    const TARGET_SYNC_EVERY: usize = 1_000;      // ← sync every 1 000 steps
    let mut step_count = 0;

    let mut target    = model.clone();           // frozen target net
    let mut optimizer = AdamConfig::new().init();
    let mut rng       = Xoshiro256PlusPlus::from_entropy();
    let mut total     = 0.0;
    let mut total_duration = std::time::Duration::new(0, 0);

    for ep in tqdm!(0..num_episodes) {
        if ep % episode_stop == 0 {
            let mean = total / episode_stop as f32;
            let mean_duration = total_duration / episode_stop as u32;
            logger.log(ep, mean, mean_duration);
            total = 0.0;
        }
        
        let eps = (1.0 - ep as f32 / num_episodes as f32) * start_epsilon
            + (ep as f32 / num_episodes as f32) * final_epsilon;
        
        let mut env = Env::default();
        let mut s = env.state_description();

        let game_start = Instant::now();
        while !env.is_game_over() {
            step_count += 1;
            let s_t = Tensor::<B,1>::from_floats(s.as_slice(), device);
            let mask_t = Tensor::<B,1>::from(env.action_mask()).to_device(device);
            let q_s = model.forward(s_t.clone());
            let a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s, &mask_t, minus_one, plus_one, fmin_vec,
                env.available_actions_ids(), eps, &mut rng);


            let r_prev = env.score();
            env.step_from_idx(a);
            let r     = env.score() - r_prev;
            let s2    = env.state_description();

            let y = if env.is_game_over() {
                Tensor::<B,1>::from([r]).to_device(device)
            } else {
                let s2_t = Tensor::<B,1>::from_floats(s2.as_slice(), device);
                let a_max = model.forward(s2_t.clone()).argmax(0).into_scalar() as usize;
                let q_eval = target
                    .forward(s2_t)
                    .slice([a_max..a_max + 1])
                    .detach();                
                q_eval.mul_scalar(gamma).add_scalar(r)
            };

            let q_sa = q_s.slice([a..a + 1]);
            let loss  = (q_sa - y).powf_scalar(2.0);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(alpha.into(), model, grads);

            if step_count % TARGET_SYNC_EVERY == 0 {
                target = model.clone();
            }

            s = s2;
        }
        total += env.score();
        total_duration += game_start.elapsed();
    }

    logger.save_model(&model, num_episodes);
    println!("Mean Score : {:.3}", total / episode_stop as f32);
    model
}

/// Run Double Deep Q‑Learning and then test.
pub fn run_double_deep_q_learning<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(env_name: &str,params: &DeepLearningParams) {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let minus_one = Tensor::from_floats([-1.0; NUM_ACTIONS], &device);
    let plus_one = Tensor::from_floats([1.0; NUM_ACTIONS], &device);
    let fmin_vec = Tensor::from_floats([f32::MIN; NUM_ACTIONS], &device);

    let name = format!("./data/{}/ddql", env_name);
    let mut logger = DoubleDqnLogger::new(&name, env_name.parse().unwrap(), &params);
    
    let trained = episodic_double_deep_q_learning::<
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
        params.alpha,
        params.start_epsilon,
        params.final_epsilon,
        &minus_one,
        &plus_one,
        &fmin_vec,
        params.opt_weight_decay_penalty,
        &device,
        &mut logger
    );

    if params.run_test {
        test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
    }
}
