use std::fmt::Display;
use crate::environments::env::DeepDiscreteActionsEnv;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice, EXPORT_AT_EP};
use crate::services::algorithms::exports::model_free::sarsa::SarsaLogger;
use crate::services::algorithms::helpers::{epsilon_greedy_action, get_device, test_trained_model};
use crate::services::algorithms::model::{Forward, MyQmlp};


pub fn episodic_semi_gradient_sarsa<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,

    M: Forward<B=B> + AutodiffModule<B>,
    B: AutodiffBackend<FloatElem=f32, IntElem=i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(
    mut model: M,
    num_episodes: usize,
    episode_stop:usize,
    gamma: f32,
    alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    minus_one: &Tensor<B, 1>,
    plus_one:  &Tensor<B, 1>,
    fmin_vec:  &Tensor<B, 1>,
    _weight_decay: f32,
    device: &B::Device,
    logger: &mut SarsaLogger) -> M
where
    M::InnerModule: Forward<B=B::InnerBackend>,
{
    let mut optimizer = AdamConfig::new()
        //.with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();

    let mut rng = Xoshiro256PlusPlus::from_entropy();

    let mut total_score = 0.0;
    let mut env = Env::default();
    env.set_against_random();
    
    for ep_id in tqdm!(0..=num_episodes) {
        let progress = ep_id as f32 / num_episodes as f32;
        let decayed_epsilon = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        if ep_id % episode_stop == 0 {
            let mean = total_score / episode_stop as f32;
            logger.log(ep_id, mean);
            if EXPORT_AT_EP.contains(&ep_id) {
                logger.save_model(&model, ep_id);
            }
            total_score = 0.0;
        }
        env.reset();

        if env.is_game_over() {
            continue;
        }

        let s = env.state_description();
        let s_tensor: Tensor<B, 1> = Tensor::from_floats(s.as_slice(), device);

        let mask = env.action_mask();
        let mask_tensor: Tensor<B, 1> = Tensor::from(mask).to_device(device);
        let mut q_s = model.forward(s_tensor);

        let mut a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
            &q_s,
            &mask_tensor,
            &minus_one,
            &plus_one,
            &fmin_vec,
            env.available_actions_ids(),
            decayed_epsilon,
            &mut rng
        );

        while !env.is_game_over() {
            let prev_score = env.score();
            env.step_from_idx(a);
            let r = env.score() - prev_score;

            let s_p = env.state_description();
            let s_p_tensor: Tensor<B, 1> = Tensor::from_floats(s_p.as_slice(), device);

            let mask_p = env.action_mask();
            let mask_p_tensor: Tensor<B, 1> = Tensor::from(mask_p).to_device(device);
            let q_s_p = Tensor::from_inner(model.valid().forward(s_p_tensor.clone().inner()));

            let (a_p, q_s_p_a_p) = if env.is_game_over() {
                (0, Tensor::from([0f32]).to_device(device))
            } else {
                let a_p = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                    &q_s_p,
                    &mask_p_tensor,
                    &minus_one,
                    &plus_one,
                    &fmin_vec,
                    env.available_actions_ids(),
                    decayed_epsilon,
                    &mut rng
                );
                #[allow(clippy::single_range_in_vec_init)]
                let q_s_p_a_p = q_s_p.clone().slice([a_p..(a_p + 1)]);
                (a_p, q_s_p_a_p)
            };
            let q_s_p_a_p = q_s_p_a_p.detach();
            #[allow(clippy::single_range_in_vec_init)]
            let q_s_a = q_s.clone().slice([a..(a + 1)]);

            let loss = (q_s_a - q_s_p_a_p.mul_scalar(gamma).add_scalar(r)).powf_scalar(2f32);
            let grad_loss = loss.backward();
            let grads = GradientsParams::from_grads(grad_loss, &model);

            model = optimizer.step(alpha.into(), model, grads);

            q_s = model.forward(s_p_tensor);
            a = a_p;
        }
        total_score += env.score();
    }
    println!("Mean Score : {:.3}", total_score / episode_stop as f32);
    model
}


pub fn run_episodic_semi_gradient_sarsa<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>()
{
    // Set the device for training
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    // Create the model
    let model = MyQmlp::<MyAutodiffBackend>::new(&device,
                                                 NUM_STATE_FEATURES,
                                                 NUM_ACTIONS);


    let minus_one: Tensor<MyAutodiffBackend, 1> = Tensor::from_floats([-1.0; NUM_ACTIONS], &device);
    let plus_one: Tensor<MyAutodiffBackend, 1> = Tensor::from_floats([ 1.0; NUM_ACTIONS], &device);
    let fmin_vec: Tensor<MyAutodiffBackend, 1> = Tensor::from_floats([f32::MIN; NUM_ACTIONS], &device);
    let parameters = DeepLearningParams::default();
    let mut logger = SarsaLogger::new("./data/sarsa", &parameters);
    
    // Train the model
    let model =
        episodic_semi_gradient_sarsa::<
            NUM_STATE_FEATURES,
            NUM_ACTIONS,
            _,
            MyAutodiffBackend,
            Env
        >(
            model,
            parameters.num_episodes,
            parameters.episode_stop,
            parameters.gamma,
            parameters.alpha,
            parameters.start_epsilon,
            parameters.final_epsilon,
            &minus_one,
            &plus_one,
            &fmin_vec,
            parameters.opt_weight_decay_penalty,
            &device,
            &mut logger,
        );

    // Let's play some games (press enter to show the next game)
    test_trained_model::<
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
        Env
    >(&device, model);
}
