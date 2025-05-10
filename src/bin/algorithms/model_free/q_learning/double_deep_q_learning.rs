use burn::module::AutodiffModule;
use burn::optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{epsilon_greedy_action, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::SeedableRng;

/// Double Deep Q‑Learning without experience replay.
fn episodic_double_deep_q_learning<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default,
>(
    mut model: M,
    num_episodes: usize,
    episode_stop:usize,
    gamma: f32,
    alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    minus_one: &Tensor<B, 1>,
    plus_one: &Tensor<B, 1>,
    fmin_vec: &Tensor<B, 1>,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    // target network for evaluation
    let mut target = model.clone();
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total_score = 0.0;

    for ep in tqdm!(0..num_episodes) {
        if ep > 0 && ep % episode_stop == 0 {
            println!("Mean Score : {:.3}", total_score / episode_stop as f32);
            total_score = 0.0;
        }
        let eps = {
            let frac = ep as f32 / num_episodes as f32;
            (1.0 - frac) * start_epsilon + frac * final_epsilon
        };

        let mut env = Env::default();
        env.set_against_random();
        env.reset();
        let mut s = env.state_description();

        while !env.is_game_over() {
            // 1) pick action with online network
            let s_t = Tensor::<B, 1>::from_floats(s.as_slice(), device);
            let mask_t = Tensor::<B, 1>::from(env.action_mask()).to_device(device);
            let q_s = model.forward(s_t.clone());
            let a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s, &mask_t, minus_one, plus_one, fmin_vec,
                env.available_actions_ids(), eps, &mut rng
            );

            // 2) step env
            let prev = env.score();
            env.step_from_idx(a);
            let r = env.score() - prev;
            let done = env.is_game_over();
            let s2 = env.state_description();

            // 3) compute target y = r + γ * Q_target(s', argmax_a Q_online(s', a))
            let y = if done {
                Tensor::from([r]).to_device(device)
            } else {
                let s2_t = Tensor::<B, 1>::from_floats(s2.as_slice(), device);
                // action selection by online
                let q_next_online = model.forward(s2_t.clone());
                let a_max = q_next_online.argmax(0).into_scalar() as usize;
                // evaluation by target
                let q_next_target = target.forward(s2_t);
                let q_eval = q_next_target.clone().slice([a_max..a_max + 1]);
                q_eval.mul_scalar(gamma).add_scalar(r)
            };

            // 4) current Q(s,a)
            let q_sa = q_s.clone().slice([a..a + 1]);

            // 5) loss & update online
            let loss = (q_sa - y).powf_scalar(2.0);
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(alpha.into(), model, grads);

            // 6) sync target periodically or every step
            // here: sync each episode end or each step as desired
            target = model.clone();

            s = s2;
        }
        total_score += env.score();
    }
    println!("Mean Score : {:.3}", total_score / episode_stop as f32);
    model
}

/// Run Double Deep Q‑Learning and then test.
pub fn run_double_deep_q_learning<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>() {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let minus_one = Tensor::from_floats([-1.0; NUM_ACTIONS], &device);
    let plus_one = Tensor::from_floats([1.0; NUM_ACTIONS], &device);
    let fmin_vec = Tensor::from_floats([f32::MIN; NUM_ACTIONS], &device);

    let params = DeepLearningParams::default();
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
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
