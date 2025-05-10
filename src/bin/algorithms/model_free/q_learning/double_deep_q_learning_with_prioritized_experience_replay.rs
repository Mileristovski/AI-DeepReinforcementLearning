use burn::module::AutodiffModule;
use burn::optim::{Optimizer, decay::WeightDecayConfig, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{epsilon_greedy_action, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;


struct PrioritizedReplayBuffer<S, const N: usize> {
    storage: Vec<(S, usize, f32, S, bool)>,
    priorities: Vec<f32>,
    pos: usize,
    alpha: f32,
}

impl<S: Clone, const N: usize> PrioritizedReplayBuffer<S, N> {
    fn new(alpha: f32) -> Self {
        Self { storage: Vec::with_capacity(N), priorities: Vec::with_capacity(N), pos: 0, alpha }
    }
    fn push(&mut self, transition: (S, usize, f32, S, bool)) {
        let priority = self.priorities.get(self.pos).cloned().unwrap_or(1.0);
        if self.storage.len() < N {
            self.storage.push(transition);
            self.priorities.push(priority);
        } else {
            self.storage[self.pos] = transition;
            self.priorities[self.pos] = priority;
        }
        self.pos = (self.pos + 1) % N;
    }
    /// sample indices and transitions with probability ∝ priority^alpha
    fn sample_batch(&self, batch_size: usize) -> (Vec<usize>, Vec<(S, usize, f32, S, bool)>) {
        let probs: Vec<f32> = self.priorities.iter().map(|p| p.powf(self.alpha)).collect();
        let dist = WeightedIndex::new(&probs).unwrap();
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        let mut indices = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            indices.push(dist.sample(&mut rng));
        }
        let batch = indices.iter().map(|&i| self.storage[i].clone()).collect();
        (indices, batch)
    }
    /// update priorities at given indices
    fn update_priorities(&mut self, indices: &[usize], errors: &[f32]) {
        for (&i, &err) in indices.iter().zip(errors.iter()) {
            self.priorities[i] = err.abs() + 1e-6;
        }
    }
}

/// Double DQN with prioritized experience replay
pub fn episodic_double_deep_q_learning_per<
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
    per_alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    minus_one: &Tensor<B, 1>,
    plus_one:  &Tensor<B, 1>,
    fmin_vec:  &Tensor<B, 1>,
    weight_decay: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    const REPLAY_CAPACITY: usize = 100_000;
    const BATCH_SIZE: usize = 32;
    const TARGET_UPDATE_EVERY: usize = 1_000;

    let mut target = model.clone();
    let mut buffer = PrioritizedReplayBuffer::<[f32; NUM_STATE_FEATURES], REPLAY_CAPACITY>::new(per_alpha);
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut total = 0.0;

    for ep in tqdm!(0..num_episodes) {
        if ep > 0 && ep % episode_stop == 0 {
            println!("Mean Score : {:.3}", total / episode_stop as f32);
            total = 0.0;
        }
        let eps = (1.0 - ep as f32 / num_episodes as f32) * start_epsilon
            + (ep as f32 / num_episodes as f32) * final_epsilon;

        let mut env = Env::default(); env.set_against_random(); env.reset();
        let mut s = env.state_description();

        while !env.is_game_over() {
            // choose and step
            let s_t = Tensor::<B,1>::from_floats(s.as_slice(), device);
            let mask_t = Tensor::<B,1>::from(env.action_mask()).to_device(device);
            let q_s = model.forward(s_t.clone());
            let a = epsilon_greedy_action::<B, NUM_STATE_FEATURES, NUM_ACTIONS>(
                &q_s, &mask_t, minus_one, plus_one, fmin_vec,
                env.available_actions_ids(), eps, &mut rng);

            let prev = env.score(); env.step_from_idx(a);
            let r = env.score() - prev;
            let done = env.is_game_over();
            let s2 = env.state_description();

            buffer.push((s, a, r, s2, done));
            s = s2;

            if buffer.storage.len() >= BATCH_SIZE {
                let (idxs, batch) = buffer.sample_batch(BATCH_SIZE);
                // prepare data_s, data_next same as before
                let mut data_s = [[0.0; NUM_STATE_FEATURES]; BATCH_SIZE];
                let mut data_next = [[0.0; NUM_STATE_FEATURES]; BATCH_SIZE];
                for (i, (s_i, _, _, s2_i, _)) in batch.iter().enumerate() {
                    data_s[i].copy_from_slice(s_i.as_slice());
                    data_next[i].copy_from_slice(s2_i.as_slice());
                }
                let state_t = Tensor::<B,2>::from_data(data_s, device);
                let next_t  = Tensor::<B,2>::from_data(data_next, device);

                let q_next_online = model.forward(next_t.clone());
                let q_next_target = target.forward(next_t.clone());

                // compute targets and td errors
                let mut y = [[0.0;1]; BATCH_SIZE];
                let mut errors = Vec::with_capacity(BATCH_SIZE);
                for (j, &(_, _, r_j, _, done_j)) in batch.iter().enumerate() {
                    let q_next = if done_j {
                        0.0
                    } else {
                        let row = q_next_online.clone().slice([j..j+1]);
                        let best_a = row.argmax(1).into_scalar() as usize;
                        q_next_target.clone()
                            .slice([j..j+1, best_a..best_a+1])
                            .into_scalar()
                    };
                    let target_val = r_j + gamma * q_next;
                    y[j][0] = target_val;
                    // current Q
                    let current_q = model.forward(state_t.clone())
                        .slice([j..j+1, batch[j].1..batch[j].1+1])
                        .into_scalar();
                    errors.push(target_val - current_q);
                }
                let y_t = Tensor::<B,2>::from_data(y, device);

                // gather q_t and step
                let mut q_sa = [[0.0;1]; BATCH_SIZE];
                for (j, &(_, a_j, _, _, _)) in batch.iter().enumerate() {
                    q_sa[j][0] = model.forward(state_t.clone())
                        .slice([j..j+1, a_j..a_j+1])
                        .into_scalar();
                }
                let q_t = Tensor::<B,2>::from_data(q_sa, device);
                let loss = (q_t - y_t).powf_scalar(2.0).mean();
                let grad = loss.backward();
                let grads = GradientsParams::from_grads(grad, &model);
                model = optimizer.step(alpha.into(), model, grads);

                // update priorities
                buffer.update_priorities(&idxs, &errors);
            }
        }
        total += env.score();
        if ep % TARGET_UPDATE_EVERY == 0 { target = model.clone(); }
    }

    println!("Mean Score : {:.3}", total / episode_stop as f32);
    model
}

/// Runner for prioritized Double DQN
pub fn run_double_dqn_per<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>() {
    let device: MyDevice = get_device();
    println!("Using device: {:?}\n", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let minus_one = Tensor::from_floats([-1.0; NUM_ACTIONS], &device);
    let plus_one  = Tensor::from_floats([ 1.0; NUM_ACTIONS], &device);
    let fmin_vec  = Tensor::from_floats([f32::MIN; NUM_ACTIONS], &device);

    let params = DeepLearningParams::default();
    let trained = episodic_double_deep_q_learning_per::<
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
        params.per_alpha,
        params.start_epsilon,
        params.final_epsilon,
        &minus_one,
        &plus_one,
        &fmin_vec,
        params.opt_weight_decay_penalty,
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
