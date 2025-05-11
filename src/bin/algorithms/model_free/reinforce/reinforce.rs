use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{get_device, log_softmax, softmax, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use std::time::Instant;
use kdam::tqdm;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;
use crate::services::algorithms::exports::model_free::reinforce::reinforce::ReinforceLogger;

pub fn episodic_reinforce<
    const N_S: usize,
    const N_A: usize,
    M,
    B,
    Env,
>(
    mut model: M,
    num_episodes : usize,
    log_every    : usize,
    gamma        : f32,
    lr           : f32,
    _wd          : f32,
    device       : &B::Device,
    logger: &mut ReinforceLogger
) -> M
where
    B  : AutodiffBackend<FloatElem = f32, IntElem = i64>,
    M  : Forward<B = B> + AutodiffModule<B> + Clone,
    M::InnerModule: Forward<B = B::InnerBackend>,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Display + Default,
{
    let mut opt  = AdamConfig::new().init();
    let mut rng  = Xoshiro256PlusPlus::from_entropy();
    let mut mean = 0.0f32;
    let mut total_duration = std::time::Duration::new(0, 0);

    // “minus ∞” used for masking illegal actions
    let neg_inf = Tensor::<B,1>::from_floats([-1e9; N_A], device);

    for ep in tqdm!(0..num_episodes) {
        if ep % log_every == 0 {
            let mean_duration = total_duration / log_every as u32;
            logger.log(ep, mean, mean_duration);
            mean = 0.0;
        }

        let mut env = Env::default();

        // store (state, mask, action, reward)
        let mut traj: Vec<([f32; N_S], [f32; N_A], usize, f32)> = Vec::new();
        let mut s = env.state_description();

        //------------------ roll-out ------------------------------------
        let game_start = Instant::now();
        while !env.is_game_over() {
            let logits = model.forward(Tensor::<B,1>::from_floats(s.as_slice(), device));

            let mask   = env.action_mask();
            let mask_t = Tensor::<B,1>::from(mask).to_device(device);

            // masked soft-max
            let adj = logits.clone() * mask_t.clone()
                + neg_inf.clone() * (mask_t.clone().neg().add_scalar(1.0));
            let probs = softmax(adj);

            // safe sampling (skip the step if all probs are ~0)
            let p_vec = probs.clone().into_data().into_vec::<f32>().unwrap();
            let dist  = WeightedIndex::new(&p_vec)
                .unwrap_or_else(|_| WeightedIndex::new(&[1.0; N_A]).unwrap());
            let a = dist.sample(&mut rng);

            let prev = env.score();
            env.step_from_idx(a);
            let r = env.score() - prev;

            traj.push((s, mask, a, r));
            s = env.state_description();
        }

        //---------------- Monte-Carlo returns ---------------------------
        let mut g = 0.0f32;
        let mut returns = vec![0.0; traj.len()];
        for (idx, &(_,_,_,r)) in traj.iter().rev().enumerate() {
            g = r + gamma * g;
            returns[traj.len() - 1 - idx] = g;
        }

        //---------------- REINFORCE update ------------------------------
        for ((state, mask, action, _), g_t) in traj.into_iter().zip(returns.into_iter()) {
            let logits = model.forward(Tensor::<B,1>::from_floats(state.as_slice(), device));

            let mask_t = Tensor::<B,1>::from(mask).to_device(device);
            let adj    = logits.clone() * mask_t.clone()
                + neg_inf.clone() * (mask_t.neg().add_scalar(1.0));
            let logp   = log_softmax(adj).slice([action .. action+1]);

            let loss   = logp.mul_scalar(-g_t);          // maximise return
            let grads  = GradientsParams::from_grads(loss.backward(), &model);
            model      = opt.step(lr.into(), model, grads);
        }
        total_duration += game_start.elapsed();
        mean += env.score();
    }

    logger.save_model(&model, num_episodes);
    println!("Mean Score : {:.3}", mean / log_every as f32);
    model
}


pub fn run_reinforce<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(env_name: &str,params: DeepLearningParams) {
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    
    let name = format!("./data/reinforce/{}", env_name);
    let mut logger = ReinforceLogger::new(&name, &params);
    let trained = episodic_reinforce::<
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
        params.opt_weight_decay_penalty,
        &device,
        &mut logger
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}