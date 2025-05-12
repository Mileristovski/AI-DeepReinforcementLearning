use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{epsilon_greedy_action, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice, REPLAY_CAPACITY, BATCH_SIZE, TARGET_UPDATE_EVERY};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use std::time::Instant;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;
use crate::services::algorithms::exports::model_free::q_learning::double_deep_q_learning_with_er::DqnErLogger;

// Modified episodic_deep_q_learning to Double Deep Q-Learning with experience replay
struct ReplayBuffer<S, const N: usize> {
    storage: Vec<(S, usize, f32, S, bool)>,
    pos: usize,
}

impl<S: Clone, const N: usize> ReplayBuffer<S, N> {
    fn new() -> Self {
        Self { storage: Vec::with_capacity(N), pos: 0 }
    }
    fn push(&mut self, transition: (S, usize, f32, S, bool)) {
        if self.storage.len() < N {
            self.storage.push(transition);
        } else {
            self.storage[self.pos] = transition;
        }
        self.pos = (self.pos + 1) % N;
    }
    fn sample_batch(&self, batch_size: usize) -> Vec<(S, usize, f32, S, bool)> {
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        self.storage.iter()
            .cloned()
            .choose_multiple(&mut rng, batch_size)
    }
}


pub fn episodic_double_deep_q_learning_er<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default,
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
    logger: &mut DqnErLogger
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut target = model.clone();
    let mut buffer = ReplayBuffer::<[f32; NUM_STATE_FEATURES], REPLAY_CAPACITY>::new();
    let mut optimizer = AdamConfig::new()
        //.with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total = 0.0;
    let mut total_duration = std::time::Duration::new(0, 0);

    // wrap episodes in tqdm for progress bar
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
            // select and step
            let s_t = Tensor::<B,1>::from_floats(s.as_slice(), device);
            let mask_t = Tensor::<B,1>::from(env.action_mask()).to_device(device);
            let q_s = model.forward(s_t.clone());
            let a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s, &mask_t, minus_one, plus_one, fmin_vec,
                env.available_actions_ids(), eps, &mut rng);

            let prev = env.score();
            env.step_from_idx(a);
            let r = env.score() - prev;
            let done = env.is_game_over();
            let s2 = env.state_description();

            buffer.push((s, a, r, s2, done));
            s = s2;

            if buffer.storage.len() >= BATCH_SIZE {
                let batch = buffer.sample_batch(BATCH_SIZE);
                let mut s_buf  = [[0.0; NUM_STATE_FEATURES]; BATCH_SIZE];
                let mut s2_buf = [[0.0; NUM_STATE_FEATURES]; BATCH_SIZE];
                let mut a_buf  = [0usize; BATCH_SIZE];
                let mut r_buf  = [0.0   ; BATCH_SIZE];
                let mut d_buf  = [0.0   ; BATCH_SIZE];

                for j in 0..BATCH_SIZE {
                    let (s_, a_, r_, s2_, done_) = &batch[j];
                    s_buf [j].copy_from_slice(s_.as_slice());
                    s2_buf[j].copy_from_slice(s2_.as_slice());
                    a_buf [j] = *a_;
                    r_buf [j] = *r_;
                    d_buf [j] = if *done_ { 1.0 } else { 0.0 };
                }
                let s_t  = Tensor::<B,2>::from_data(s_buf , device);
                let s2_t = Tensor::<B,2>::from_data(s2_buf, device);

                // ── compute Double-DQN target -----------------------------------------
                let q_next_online  = model .forward(s2_t.clone());   
                let best_a         = q_next_online.argmax(1);        
                let q_next_target  = target.forward(s2_t);           
                let q_next         = q_next_target
                    .gather(1, best_a.clone())
                    .squeeze::<1>(1);                                     // [B]
                let done_mask      = Tensor::<B,1>::from_floats(d_buf, device);
                let target_vec     = Tensor::<B,1>::from_floats(r_buf, device)
                    + q_next * (done_mask.mul_scalar(-1.0).add_scalar(1.0)) * gamma; 

                // ── current Q(s,a) -----------------------------------------------------
                let q_online = model.forward(s_t);
                let mut ind_buf = [[0i64; 1]; BATCH_SIZE];
                for j in 0..BATCH_SIZE {
                    ind_buf[j][0] = a_buf[j] as i64;
                }
                let ind = Tensor::<B, 2, Int>::from_data(ind_buf, device);
                let q_sa = q_online.gather(1, ind).squeeze::<1>(1);       

                // ── loss, backward, optimise ------------------------------------------
                let td   = target_vec.clone() - q_sa;                     // [B]
                let loss = td.powf_scalar(2.0).mean();

                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(alpha.into(), model, grads);
            }
        }

        total += env.score();
        total_duration += game_start.elapsed();

        if ep % TARGET_UPDATE_EVERY == 0 {
            target = model.clone();
        }
    }

    logger.save_model(&model, num_episodes);
    println!("Mean Score : {:.3}", total / episode_stop as f32);
    model
}


/// Run Double Deep Q‑Learning and then test.
pub fn run_double_deep_q_learning_er<
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
    
    let name = format!("./data/{}/ddqler", env_name);
    let mut logger = DqnErLogger::new(&name, &params, REPLAY_CAPACITY, BATCH_SIZE, TARGET_UPDATE_EVERY);
    let trained = episodic_double_deep_q_learning_er::<
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
