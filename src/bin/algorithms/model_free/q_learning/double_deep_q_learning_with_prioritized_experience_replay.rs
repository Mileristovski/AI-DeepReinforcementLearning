use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::services::algorithms::helpers::{epsilon_greedy_action, get_device, sample_distinct_weighted, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, MyDevice, CAPACITY, BATCH, TARGET_EVERY, PRIO_EPS, PRIO_MAX, BETA_START, BETA_END, BETA_FRAMES, REPLAY_CAPACITY, BATCH_SIZE, TARGET_UPDATE_EVERY};
use crate::services::algorithms::model::{Forward, MyQmlp};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use std::time::Instant;
use kdam::tqdm;
use rand::SeedableRng;
use crate::services::algorithms::exports::model_free::q_learning::double_deep_q_learning_with_per::DqnPerLogger;

struct PrioritizedReplayBuffer<S, const N: usize> {
    storage: Vec<(S, usize, f32, S, bool)>,
    priorities: Vec<f32>,
    pos: usize
}

impl<S: Clone, const N: usize> PrioritizedReplayBuffer<S, N> {
    fn new() -> Self {
        Self { storage: Vec::with_capacity(N), priorities: Vec::with_capacity(N), pos: 0 }
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
    /// update priorities at given indices
    fn update_priorities(&mut self, indices: &[usize], errors: &[f32]) {
        for (&i, &err) in indices.iter().zip(errors.iter()) {
            self.priorities[i] = err.abs() + 1e-6;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn episodic_double_deep_q_learning_per<
    const N_S: usize,
    const N_A: usize,
    M,
    B,
    Env,
>(
    mut model: M,
    num_episodes: usize,
    log_every:    usize,
    gamma:        f32,
    lr:           f32,
    per_alpha:    f32,
    eps_start:    f32,
    eps_final:    f32,
    minus_one: &Tensor<B, 1>,
    plus_one : &Tensor<B, 1>,
    fmin_vec : &Tensor<B, 1>,
    _weight_decay: f32,
    device: &B::Device,
    logger: &mut DqnPerLogger
) -> M
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    M::InnerModule: Forward<B = B::InnerBackend>,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Display + Default,
{

    let mut target = model.clone();
    let mut opt = AdamConfig::new()
        // .with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();
    let mut buf =
        PrioritizedReplayBuffer::<[f32; N_S], CAPACITY>::new();

    let mut rng        = Xoshiro256PlusPlus::from_entropy();
    let mut score_sum  = 0.0;
    let mut total_duration = std::time::Duration::new(0, 0);
    let mut grad_steps = 0usize;

    for ep in tqdm!(0..num_episodes) {
         if ep % log_every == 0 {
            let mean = score_sum / log_every as f32;
             let mean_duration = total_duration / log_every as u32;
             logger.log(ep, mean, mean_duration);
             score_sum = 0.0;
        }
        // linear ε-decay
        let eps = eps_start * (1.0 - ep as f32 / num_episodes as f32)
            + eps_final * (ep as f32 / num_episodes as f32);

        let mut env = Env::default();
        let mut s = env.state_description();

        let game_start = Instant::now();
        while !env.is_game_over() {
            // ────────── action ──────────
            let q_s = model.forward(Tensor::<B,1>::from_floats(s.as_slice(), device));
            let a = epsilon_greedy_action::<B, N_S, N_A>(
                &q_s,
                &Tensor::from(env.action_mask()).to_device(device),
                minus_one, plus_one, fmin_vec,
                env.available_actions_ids(),
                eps, &mut rng);

            let prev = env.score();
            env.step_from_idx(a);
            let r = env.score() - prev;
            let done = env.is_game_over();
            let s2 = env.state_description();

            buf.push((s, a, r, s2, done));
            s = s2;

            // ────────── learn ───────────
            if buf.storage.len() >= BATCH {
                // (1) distinct weighted sample
                let probs: Vec<f32> = buf.priorities
                    .iter().map(|p| p.powf(per_alpha)).collect();
                let idx = sample_distinct_weighted(&probs, BATCH, &mut rng);
                let batch: Vec<_> = idx.iter().map(|&i| buf.storage[i].clone()).collect();

                // (2) IS weights
                let beta = BETA_END.min(
                    BETA_START + grad_steps as f32 / BETA_FRAMES as f32
                        * (BETA_END - BETA_START));
                let max_p = probs.iter().copied().fold(f32::MIN, f32::max);
                let is_w: Vec<f32> = idx.iter()
                    .map(|&i| (probs[i] / max_p).powf(-beta))
                    .collect();

                // (3) tensor buffers
                let mut s_buf  = [[0.0; N_S]; BATCH];
                let mut s2_buf = [[0.0; N_S]; BATCH];
                let mut a_buf  = [0usize; BATCH];
                let mut r_buf  = [0.0f32; BATCH];
                let mut d_buf  = [0.0f32; BATCH];

                for j in 0..BATCH {
                    let (s_, a_, r_, s2_, d_) = &batch[j];
                    s_buf [j].copy_from_slice(s_.as_slice());
                    s2_buf[j].copy_from_slice(s2_.as_slice());
                    a_buf [j] = *a_;
                    r_buf [j] = *r_;
                    d_buf [j] = if *d_ { 1.0 } else { 0.0 };
                }
                let s_t  = Tensor::<B,2>::from_data(s_buf , device);
                let s2_t = Tensor::<B,2>::from_data(s2_buf, device);

                // (4) Double-DQN target
                let q_next_online  = model .forward(s2_t.clone());   // [B,A]
                let best_a         = q_next_online.argmax(1);        // [B,1]
                let q_next_target  = target.forward(s2_t);           // [B,A]
                let q_next = q_next_target
                    .gather(1, best_a.clone())
                    .squeeze::<1>(1);                                   // [B]
                let done_mask = Tensor::<B,1>::from_floats(d_buf, device);
                let q_next = q_next * (done_mask.mul_scalar(-1.0).add_scalar(1.0));

                let target_vec = Tensor::<B,1>::from_floats(r_buf, device)
                    + q_next * gamma;

                // (5) current Q(s,a)
                let q_online = model.forward(s_t);
                let mut ind_buf = [[0i64; 1]; BATCH];
                for j in 0..BATCH {
                    ind_buf[j][0] = a_buf[j] as i64;
                }
                let ind = Tensor::<B, 2, Int>::from_data(ind_buf, device);
                let q_sa = q_online
                    .gather(1, ind)
                    .squeeze::<1>(1);
                let td = target_vec.clone() - q_sa;
                let w_t = Tensor::<B,1>::from_floats(is_w.as_slice(), device);
                let loss = (td.clone() * w_t).powf_scalar(2.0).mean();

                // (6) optimise
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = opt.step(lr.into(), model, grads);
                grad_steps += 1;

                // (7) update priorities
                let td_cpu = td.into_data().into_vec::<f32>().unwrap();
                let new_p: Vec<f32> = td_cpu.iter()
                    .map(|e| (e.abs() + PRIO_EPS).min(PRIO_MAX))
                    .collect();
                buf.update_priorities(&idx, &new_p);

                // (8) target net sync
                if grad_steps % TARGET_EVERY == 0 {
                    target = model.clone();
                }
            }
        }
        total_duration += game_start.elapsed();
        score_sum += env.score();
    }
    
    logger.save_model(&model, num_episodes);
    println!("Mean Score : {:.3}", score_sum / log_every as f32);
    model
}

/// Runner for prioritized Double DQN
pub fn run_double_dqn_per<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(env_name: &str,params: &DeepLearningParams) {
    let device: MyDevice = get_device();
    println!("Using device: {:?}\n", device);

    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS);
    let minus_one = Tensor::from_floats([-1.0; NUM_ACTIONS], &device);
    let plus_one  = Tensor::from_floats([ 1.0; NUM_ACTIONS], &device);
    let fmin_vec  = Tensor::from_floats([f32::MIN; NUM_ACTIONS], &device);

    let name = format!("./data/ddqlper/{}", env_name);
    let mut logger = DqnPerLogger::new(&name, &params, REPLAY_CAPACITY, BATCH_SIZE, TARGET_UPDATE_EVERY);
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
        params.ql_per_alpha,
        params.start_epsilon,
        params.final_epsilon,
        &minus_one,
        &plus_one,
        &fmin_vec,
        params.opt_weight_decay_penalty,
        &device,
        &mut logger,
    );

    if params.run_test {
        test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
    }
}
