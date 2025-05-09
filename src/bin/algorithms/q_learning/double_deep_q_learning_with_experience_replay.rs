use burn::module::AutodiffModule;
use burn::optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algo_helper::helpers::{epsilon_greedy_action, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algo_helper::qmlp::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;

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
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    const REPLAY_CAPACITY: usize = 100_000;
    const BATCH_SIZE: usize = 32;
    const TARGET_UPDATE_EVERY: usize = 1_000;

    let mut target = model.clone();
    let mut buffer = ReplayBuffer::<[f32; NUM_STATE_FEATURES], REPLAY_CAPACITY>::new();
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-7)))
        .init();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total_score = 0.0;

    // wrap episodes in tqdm for progress bar
    for ep in tqdm!(0..num_episodes) {
        // every 100 episodes, print mean score
        if ep > 0 && ep % episode_stop == 0 {
            println!("Episode {:>5}/{}, Mean Score: {:.3}", ep, num_episodes, total_score / episode_stop as f32);
            total_score = 0.0;
        }

        let eps = (1.0 - ep as f32 / num_episodes as f32) * start_epsilon
            + (ep as f32 / num_episodes as f32) * final_epsilon;

        let mut env = Env::default();
        env.set_against_random();
        env.reset();
        let mut s = env.state_description();

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
                let mut data_s = [[0.0f32; NUM_STATE_FEATURES]; BATCH_SIZE];
                let mut data_next = [[0.0f32; NUM_STATE_FEATURES]; BATCH_SIZE];
                for (i, (s_i, _, _, s2_i, _)) in batch.iter().enumerate() {
                    data_s[i].copy_from_slice(s_i.as_slice());
                    data_next[i].copy_from_slice(s2_i.as_slice());
                }
                let state_t = Tensor::<B,2>::from_data(data_s, device);
                let next_t  = Tensor::<B,2>::from_data(data_next, device);

                let q_next_online = model.forward(next_t.clone());
                let q_next_target = target.forward(next_t.clone());

                let mut y = [[0.0f32;1]; BATCH_SIZE];
                for (j, &(_, _, r_j, _, done_j)) in batch.iter().enumerate() {
                    let yj = if done_j {
                        r_j
                    } else {
                        let row = q_next_online.clone().slice([j..j+1]);
                        let best_a = row.argmax(1).into_scalar() as usize;
                        let q_val = q_next_target.clone()
                            .slice([j..j+1, best_a..best_a+1])
                            .into_scalar();
                        r_j + gamma * q_val
                    };
                    y[j][0] = yj;
                }
                let y_t = Tensor::<B,2>::from_data(y, device);

                let q_vals = model.forward(state_t.clone());
                let mut q_sa = [[0.0f32;1]; BATCH_SIZE];
                for (j, &(_, a_j, _, _, _)) in batch.iter().enumerate() {
                    q_sa[j][0] = q_vals.clone()
                        .slice([j..j+1, a_j..a_j+1])
                        .into_scalar();
                }
                let q_t = Tensor::<B,2>::from_data(q_sa, device);

                let loss = (q_t - y_t).powf_scalar(2.0).mean();
                let grad = loss.backward();
                let grads = GradientsParams::from_grads(grad, &model);
                model = optimizer.step(alpha.into(), model, grads);
            }
        }

        total_score += env.score();

        if ep % TARGET_UPDATE_EVERY == 0 {
            target = model.clone();
        }
    }

    println!("Mean Score : {:.3}", total_score / 1000.0);
    model
}


/// Run Double Deep Q‑Learning and then test.
pub fn run_double_deep_q_learning_er<
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
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
