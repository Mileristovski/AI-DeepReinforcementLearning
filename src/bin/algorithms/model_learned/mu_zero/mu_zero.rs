use burn::module::AutodiffModule;
use burn::optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;
use crate::services::algorithms::helpers::{run_mcts_pi, log_softmax, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;
use crate::services::algorithms::model::{Forward, MyQmlp};

struct ReplayBuffer<T> {
        storage: Vec<T>,
        pos: usize,
        n: usize,
    }
impl<T: Clone> ReplayBuffer<T> {
        fn new(n: usize) -> Self {
                Self { storage: Vec::with_capacity(n), pos: 0, n }
            }
        fn push(&mut self, item: T) {
                if self.storage.len() < self.n {
                    self.storage.push(item);
                } else {
                    self.storage[self.pos] = item;
                }
                self.pos = (self.pos + 1) % self.n;
            }
        fn sample_batch(&self, batch_size: usize) -> Vec<T> {
                let mut rng = Xoshiro256PlusPlus::from_entropy();
                self.storage.iter().cloned().choose_multiple(&mut rng, batch_size)
            }
    }

pub fn episodic_mu_zero<
    const HS: usize,
    const A: usize,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<HS, A> + Display + Default + Clone,
>(
    mut model: M,
    num_episodes: usize,
    episode_stop: usize,
    replay_cap: usize,
    batch_size: usize,
    games_per_iter: usize,
    mcts_sims: usize,
    learning_rate: f32,
    c: f32,
    weight_decay_penalty: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(weight_decay_penalty)))
        .init();
    
    let mut buffer = ReplayBuffer::<([f32; HS], [f32; A], f32)>::new(replay_cap);
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut history: [f32; HS];
    let mut total = 0.0;

    for _iter in tqdm!(0..num_episodes) {
        if _iter > 0 && _iter % episode_stop == 0 {
            println!("Mean Score : {:.3}", total / episode_stop as f32);
            total = 0.0;
        }
        for _ in 0..games_per_iter {
            let mut env = Env::default();
            env.set_against_random();
            env.reset();
            let mut traj = Vec::new();

            while !env.is_game_over() {
                history = env.state_description();

                let pi_root: [f32; A] = run_mcts_pi::<HS, A, Env, _>(
                    &env,
                    mcts_sims,
                    c,
                    &mut rng,
                );

                // sample action
                let dist = WeightedIndex::new(&pi_root).unwrap();
                let a = dist.sample(&mut rng);

                traj.push((history, pi_root));
                env.step_from_idx(a);
            }
            let z = env.score().signum();
            for (s_hist, pi_root) in traj {
                buffer.push((s_hist, pi_root, z));
            }
            total += env.score();
        }

        // ==== training ====
        if buffer.storage.len() >= batch_size {
            let batch = buffer.sample_batch(batch_size);
            let mut loss_tot = Tensor::<B, 1>::from([0.0f32]).to_device(device);

            for (s_hist, pi_target, z) in batch {
                let x = Tensor::<B, 1>::from_floats(s_hist, device);

                // forward pass
                let out = model.forward(x);
                let pi_logits = out.clone().slice([0..A]);
                let v_pred    = out.clone().slice([A..A+1]);
                let r_pred    = out.clone().slice([A+1..A+2]);

                // policy loss
                let pi_t = Tensor::<B, 1>::from(pi_target).to_device(device);
                let logp = log_softmax(pi_logits);
                let loss_p = -(pi_t * logp).sum();

                // value loss
                let z_t = Tensor::<B, 1>::from_floats([z], device);
                let loss_v = (v_pred - z_t.clone()).powf_scalar(2.0);

                // reward loss
                let loss_r = (r_pred - z_t).powf_scalar(2.0);

                loss_tot = loss_tot + loss_p + loss_v + loss_r;
            }

            let loss = loss_tot.clone() / (batch_size as f32);
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(learning_rate.into(), model, grads);
        }
    }
    println!("Mean Score : {:.3}", total / (episode_stop as f32));
    model
}

pub fn run_mu_zero<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>() {
    let device = get_device();
    println!("Using device: {:?}", device);

    // network outputs A+1 dims: [policy_logits(A), value(1), reward(1)]
    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS + 1);

    let params = DeepLearningParams::default();
    let trained = episodic_mu_zero::<NUM_STATE_FEATURES, NUM_ACTIONS, _, _, Env>(
        model,
        params.num_episodes,
        params.episode_stop,
        params.mz_games_per_iter,
        params.mz_replay_cap,
        params.mz_batch_size,
        params.mcts_simulations,
        params.policy_lr,
        params.mz_c,
        params.mz_weight_decay_penalty,
        &device,
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
