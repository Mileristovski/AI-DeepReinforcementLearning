use burn::module::AutodiffModule;
use burn::optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;
use crate::services::algo_helper::helpers::{run_mcts_pi, log_softmax, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;
use crate::services::algo_helper::qmlp::{Forward, MyQmlp};

struct ReplayBuffer<T, const N: usize> {
        storage: Vec<T>,
        pos: usize,
    }
impl<T: Clone, const N: usize> ReplayBuffer<T, N> {
        fn new() -> Self {
                Self { storage: Vec::with_capacity(N), pos: 0 }
            }
        fn push(&mut self, item: T) {
                if self.storage.len() < N {
                    self.storage.push(item);
                } else {
                    self.storage[self.pos] = item;
                }
                self.pos = (self.pos + 1) % N;
            }
        fn sample_batch(&self, batch_size: usize) -> Vec<T> {
                let mut rng = Xoshiro256PlusPlus::from_entropy();
                self.storage.iter().cloned().choose_multiple(&mut rng, batch_size)
            }
    }

/// “Flattened‑history” MuZero (no dynamics unroll, K=1)
pub fn episodic_mu_zero<
    const HS: usize,     // history length × state_dim
    const A: usize,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<HS, A> + Display + Default + Clone,
>(
    mut model: M,
    num_episodes: usize,
    games_per_iter: usize,
    mcts_sims: usize,
    gamma: f32,
    learning_rate: f32,
    device: &B::Device,
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .init();

    const REPLAY_CAP: usize = 100_000;
    const BATCH_SIZE: usize = 256;
    let mut buffer = ReplayBuffer::<([f32; HS], [f32; A], f32), REPLAY_CAP>::new();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut history: [f32; HS] = [0.0; HS];

    for _iter in tqdm!(0..num_episodes) {
        // ==== self‑play collection ====
        for _ in 0..games_per_iter {
            let mut env = Env::default();
            env.set_against_random();
            env.reset();
            let mut traj = Vec::new();

            while !env.is_game_over() {
                // record flattened “history”
                history = env.state_description();

                // run MCTS‐π at root (4 args: env, sims, c, rng)
                let pi_root: [f32; A] = run_mcts_pi::<HS, A, Env, _>(
                    &env,
                    mcts_sims,
                    /* c = */ 1.0,
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
        }

        // ==== training ====
        if buffer.storage.len() >= BATCH_SIZE {
            let batch = buffer.sample_batch(BATCH_SIZE);
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

            let loss = loss_tot.clone() / (BATCH_SIZE as f32);
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(learning_rate.into(), model, grads);
        }
    }

    model
}

/// Train + basic test loop for MuZero
pub fn run_mu_zero<
    const HS: usize,
    const A: usize,
    Env: DeepDiscreteActionsEnv<HS, A> + Display + Default + Clone,
>() {
    let device = get_device();

    // network outputs A+1 dims: [policy_logits(A), value(1), reward(1)]
    let model = MyQmlp::<MyAutodiffBackend>::new(&device, HS, A + 1);

    let params = DeepLearningParams::default();
    let trained = episodic_mu_zero::<HS, A, _, _, Env>(
        model,
        /*num_episodes=*/5_000,
        /*games_per_iter=*/1,
        /*mcts_sims=*/params.mcts_simulations,
        /*gamma=*/params.gamma,
        /*learning_rate=*/params.policy_lr,
        &device,
    );

    test_trained_model::<HS, A, Env>(&device, trained);
}
