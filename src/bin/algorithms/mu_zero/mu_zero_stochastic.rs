use burn::module::AutodiffModule;
use burn::optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::distributions::{Distribution,WeightedIndex};
use rand_distr::StandardNormal;
use crate::services::algo_helper::helpers::{run_mcts_pi, log_softmax, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand_xoshiro::rand_core::SeedableRng;
use crate::services::algo_helper::qmlp::{Forward, MyQmlp};


struct ReplayBuffer<T, const N: usize> {
    storage: Vec<T>,
    pos: usize,
}
impl<T: Clone, const N: usize> ReplayBuffer<T, N> {
    fn new() -> Self { Self { storage: Vec::with_capacity(N), pos: 0 } }
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

/// Stochastic MuZero: latent state is Gaussian.  We add a KL term pushing (μ,σ)→N(0,I).
pub fn episodic_mu_zero_stochastic<
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
        // self‑play
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
                    /* c = */ 1.0,
                    &mut rng,
                );
                let dist = WeightedIndex::new(&pi_root).unwrap();
                let a = dist.sample(&mut rng);

                traj.push((history, pi_root));
                env.step_from_idx(a);
            }
            let z = env.score().signum();
            for (h, pi) in traj {
                buffer.push((h, pi, z));
            }
        }

        // train
        if buffer.storage.len() >= BATCH_SIZE {
            let batch = buffer.sample_batch(BATCH_SIZE);
            let mut loss_tot = Tensor::<B, 1>::from([0.0f32]).to_device(device);

            for (h, pi_target, z) in batch {
                // 1) representation from flat history
                let x = Tensor::<B, 1>::from_floats(h, device);
                let latent_params = model.forward(x.clone());
                // split into: [policy_logits(A), value(1), reward(1), μ(HS), logvar(HS)]
                let policy_logits = latent_params.clone().slice([0..A]);
                let v_pred        = latent_params.clone().slice([A..A+1]);
                let r_pred        = latent_params.clone().slice([A+1..A+2]);
                let μ             = latent_params.clone().slice([A+2..A+2+HS]);
                let logvar        = latent_params.slice([A+2+HS..A+2+2*HS]);

                // sample stochastic latent (for dynamics you might feed this back in; here just use for loss)
                let mut eps_arr: [f32; HS] = [0.0; HS];
                for x in eps_arr.iter_mut() {
                    // Option A: bind a f32
                    let sample: f32 = StandardNormal.sample(&mut rng);
                    *x = sample;
                }
                
                // turn that into a tensor
                let ϵ = Tensor::<B, 1>::from_floats(eps_arr, device);

                let σ = logvar.clone().mul_scalar(0.5).exp();
                let z_latent = μ.clone() + σ.clone() * ϵ;

                // policy loss
                let pi_t = Tensor::<B, 1>::from(pi_target).to_device(device);
                let logp = log_softmax(policy_logits);
                let loss_p = -(pi_t * logp).sum();

                // value & reward losses
                let z_t = Tensor::<B, 1>::from_floats([z], device);
                let loss_v = (v_pred - z_t.clone()).powf_scalar(2.0);
                let loss_r = (r_pred - z_t.clone()).powf_scalar(2.0);

                // KL( N(μ,σ²) ∥ N(0,1) ) = ½ sum(μ² + σ² − logσ² −1)
                let kl = μ.clone().powf_scalar(2.0)
                    + σ.clone().powf_scalar(2.0)
                    - logvar.clone()
                    - Tensor::from_floats([1.0f32; HS], device);
                let loss_kl = kl.sum().mul_scalar(0.5);

                loss_tot = loss_tot + loss_p + loss_v + loss_r + loss_kl;
            }

            let loss = loss_tot.clone() / (BATCH_SIZE as f32);
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(learning_rate.into(), model, grads);
        }
    }

    model
}

/// Runner for stochastic MuZero
pub fn run_muzero_stochastic<
    const HS: usize,
    const A: usize,
    Env: DeepDiscreteActionsEnv<HS, A> + Display + Default + Clone,
>() {
    let device = get_device();
    // now network outputs A + 1 + 1 + HS + HS dims
    let model = MyQmlp::<MyAutodiffBackend>::new(&device, HS, A+1+1+HS+HS);

    let params = DeepLearningParams::default();
    let trained = episodic_mu_zero_stochastic::<HS, A, _, _, Env>(
        model,
        params.num_episodes,
        params.episode_stop,        // games_per_iter
        params.mcts_simulations,
        params.gamma,
        params.policy_lr,
        &device,
    );

    test_trained_model::<HS, A, Env>(&device, trained);
}
