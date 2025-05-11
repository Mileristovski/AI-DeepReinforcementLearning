use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::distributions::{Distribution,WeightedIndex};
use rand_distr::StandardNormal;
use crate::services::algorithms::helpers::{run_mcts_pi, log_softmax, get_device, test_trained_model};
use crate::config::{DeepLearningParams, MyAutodiffBackend, EXPORT_AT_EP};
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand_xoshiro::rand_core::SeedableRng;
use crate::services::algorithms::exports::model_learned::mu_zero::mu_zero_sto::MuZeroStochasticLogger;
use crate::services::algorithms::model::{Forward, MyQmlp};


struct ReplayBuffer<T> {
    storage: Vec<T>,
    pos: usize,
    n: usize,
}
impl<T: Clone> ReplayBuffer<T> {
    fn new(n: usize) -> Self { Self { storage: Vec::with_capacity(n), pos: 0, n } }
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

pub fn episodic_mu_zero_stochastic<
    const HS: usize,     // history length × state_dim
    const A: usize,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    Env: DeepDiscreteActionsEnv<HS, A> + Display + Default + Clone,
>(
    mut model: M,
    num_episodes: usize,
    episode_stop: usize,
    games_per_iter: usize,
    mcts_sims: usize,
    learning_rate: f32,
    c: f32,
    replay_cap: usize,
    batch_size: usize,
    _weight_decay:      f32,
    device: &B::Device,
    logger: &mut MuZeroStochasticLogger
) -> M
where
    M::InnerModule: Forward<B = B::InnerBackend>,
{
    let mut optimizer = AdamConfig::new()
        //.with_weight_decay(Some(WeightDecayConfig::new(_weight_decay)))
        .init();

    let mut buffer = ReplayBuffer::<([f32; HS], [f32; A], f32)>::new(replay_cap);
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut history: [f32; HS];
    let mut total = 0.0;
    let mut n_games= 1usize;
    
    for _iter in tqdm!(0..=num_episodes) {
        if _iter % episode_stop == 0 {
            let mean = total / n_games as f32;
            logger.log(_iter, mean);
            if EXPORT_AT_EP.contains(&_iter) {
                logger.save_model(&model, _iter);
            }
            total = 0.0;
        }

        for _ in 0..games_per_iter {
            let mut env = Env::default();
            let mut traj = Vec::new();

            while !env.is_game_over() {
                history = env.state_description();

                let pi_root: [f32; A] = run_mcts_pi::<HS, A, Env, _>(
                    &env,
                    mcts_sims,
                    c,
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
            total += env.score();
            n_games    += 1;
        }

        // train
        if buffer.storage.len() >= batch_size {
            let batch = buffer.sample_batch(batch_size);
            let mut loss_tot = Tensor::<B, 1>::from([0.0f32]).to_device(device);

            for (h, pi_target, z) in batch {
                // 1) representation from flat history
                let x = Tensor::<B, 1>::from_floats(h, device);
                let latent_params = model.forward(x.clone());
                
                let policy_logits = latent_params.clone().slice([0..A]);
                let v_pred        = latent_params.clone().slice([A..A+1]);
                let r_pred        = latent_params.clone().slice([A+1..A+2]);
                let mu             = latent_params.clone().slice([A+2..A+2+HS]);
                let logvar        = latent_params.slice([A+2+HS..A+2+2*HS]);

                let mut eps_arr: [f32; HS] = [0.0; HS];
                for x in eps_arr.iter_mut() {
                    // Option A: bind a f32
                    let sample: f32 = StandardNormal.sample(&mut rng);
                    *x = sample;
                }
                
                // turn that into a tensor
                // let ϵ = Tensor::<B, 1>::from_floats(eps_arr, device);
                let sigma = logvar.clone().mul_scalar(0.5).exp();
                // let z_latent = mu.clone() + sigma.clone() * ϵ;

                // policy loss
                let pi_t = Tensor::<B, 1>::from(pi_target).to_device(device);
                let logp = log_softmax(policy_logits);
                let loss_p = -(pi_t * logp).sum();

                // value & reward losses
                let z_t = Tensor::<B, 1>::from_floats([z], device);
                let loss_v = (v_pred - z_t.clone()).powf_scalar(2.0);
                let loss_r = (r_pred - z_t.clone()).powf_scalar(2.0);

                // KL( N(mu,sigma²) ∥ N(0,1) ) = ½ sum(mu² + sigma² − logsigma² −1)
                let kl = mu.clone().powf_scalar(2.0)
                    + sigma.clone().powf_scalar(2.0)
                    - logvar.clone()
                    - Tensor::from_floats([1.0f32; HS], device);
                let loss_kl = kl.sum().mul_scalar(0.5);

                loss_tot = loss_tot + loss_p + loss_v + loss_r + loss_kl;
            }

            let loss = loss_tot.clone() / (batch_size as f32);
            let grad = loss.backward();
            let grads = GradientsParams::from_grads(grad, &model);
            model = optimizer.step(learning_rate.into(), model, grads);
        }
    }
    
    println!("Mean Score : {:.3}", total / (n_games as f32));
    model
}

pub fn run_muzero_stochastic<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>() {
    let device = get_device();
    println!("Using device: {:?}", device);
    
    let model = MyQmlp::<MyAutodiffBackend>::new(&device, NUM_STATE_FEATURES, NUM_ACTIONS+1+1+NUM_STATE_FEATURES+NUM_STATE_FEATURES);

    let params = DeepLearningParams::default();
    let mut logger = MuZeroStochasticLogger::new("./data/mu_zero_sto", &params);
    let trained = episodic_mu_zero_stochastic::<NUM_STATE_FEATURES, NUM_ACTIONS, _, _, Env>(
        model,
        params.num_episodes,
        params.episode_stop,
        params.mcts_simulations,
        params.mz_games_per_iter,
        params.ac_policy_lr,
        params.mz_c,
        params.mz_replay_cap,
        params.mz_batch_size,
        params.opt_weight_decay_penalty,
        &device,
        &mut logger
    );

    test_trained_model::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(&device, trained);
}
