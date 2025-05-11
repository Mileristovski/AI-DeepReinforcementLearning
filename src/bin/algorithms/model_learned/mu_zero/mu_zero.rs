use burn::module::AutodiffModule;
use burn::optim::{Optimizer, GradientsParams, AdamConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use kdam::tqdm;
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::{IteratorRandom, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::fmt::Display;

use crate::config::{DeepLearningParams, MyAutodiffBackend, EXPORT_AT_EP};
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::services::algorithms::exports::model_learned::mu_zero::mu_zero::MuZeroLogger;
use crate::services::algorithms::helpers::{run_mcts_pi, get_device, test_trained_model, masked_log_softmax};
use crate::services::algorithms::model::{Forward, MyQmlp};

// ─────────────────────────────────────────────────────────────
//  Simple circular replay buffer
// ─────────────────────────────────────────────────────────────
struct ReplayBuffer<T> {
    storage: Vec<T>,
    pos:     usize,
    cap:     usize,
}
impl<T: Clone> ReplayBuffer<T> {
    fn new(cap: usize) -> Self { Self { storage: Vec::with_capacity(cap), pos: 0, cap } }
    fn push(&mut self, item: T) {
        if self.storage.len() < self.cap { self.storage.push(item); }
        else { self.storage[self.pos] = item; }
        self.pos = (self.pos + 1) % self.cap;
    }
    fn sample_batch(&self, n: usize) -> Vec<T> {
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        self.storage.iter().cloned().choose_multiple(&mut rng, n)
    }
}

// ─────────────────────────────────────────────────────────────
//  MuZero training loop (minimal)
// ─────────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
pub fn episodic_mu_zero<
    const HS: usize,
    const A : usize,
    M, B, Env
>(
    mut model: M,
    num_episodes:      usize,
    episode_stop:      usize,
    replay_cap:        usize,
    batch_size:        usize,
    games_per_iter:    usize,
    mcts_sims:         usize,
    learning_rate:     f32,
    c:                 f32,
    _weight_decay:      f32,
    device:            &B::Device,
    logger: &mut MuZeroLogger
) -> M
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i64>,
    M: Forward<B = B> + AutodiffModule<B> + Clone,
    M::InnerModule: Forward<B = B::InnerBackend>,
    Env: DeepDiscreteActionsEnv<HS, A> + Display + Default + Clone,
{
    // optimiser ------------------------------------------------------------
    let mut opt = AdamConfig::new()
        //.with_weight_decay(Some(WeightDecayConfig::new(weight_decay)))
        .init();

    // replay buffer & rng --------------------------------------------------
    let mut rb  = ReplayBuffer::<([f32; HS], [f32; A], f32)>::new(replay_cap);
    let mut rng = Xoshiro256PlusPlus::from_entropy();

    // bookkeeping ----------------------------------------------------------
    let mut total_score  = 0.0f32;   // sum over *games*
    let mut games_count  = 0usize;   // #games so far in current block

    // training loop --------------------------------------------------------
    for ep in tqdm!(0..=num_episodes) {
        // ── Self‑play block ────────────────────────────────────────────
        for _ in 0..games_per_iter {
            let mut env   = Env::default();

            let mut traj = Vec::new();
            while !env.is_game_over() {
                let history = env.state_description();

                let pi_root: [f32; A] = run_mcts_pi::<HS, A, Env, _>(&env, mcts_sims, c, &mut rng);
                let dist = WeightedIndex::new(&pi_root).expect("pi not a prob‑vector");

                let a    = dist.sample(&mut rng);

                traj.push((history, pi_root));
                env.step_from_idx(a);
            }
            let z = env.score();                 // raw final score ------------
            for (h, pi_root) in traj { rb.push((h, pi_root, z)); }

            total_score += z;
            games_count += 1;
        }

        // every episode_stop iters → print true mean and reset -------------
        if ep % episode_stop == 0 {
            let mean = total_score / episode_stop as f32;
            logger.log(ep, mean);
            if EXPORT_AT_EP.contains(&ep) {
                logger.save_model(&model, ep);
            }
            total_score = 0.0;
        }

        // ── Network update from replay buffer ────────────────────────────
        if rb.storage.len() >= batch_size {
            let batch      = rb.sample_batch(batch_size);
            let mut loss_t = Tensor::<B, 1>::from([0.0]).to_device(device);

            for (s_hist, pi_target, z) in batch {
                let x  = Tensor::<B, 1>::from_floats(s_hist, device);
                let out = model.forward(x);

                let pi_logits = out.clone().slice([0..A]);
                let v_pred    = out.clone().slice([A..A + 1]);
                let r_pred    = out.slice([A + 1..A + 2]);

                let pi_t  = Tensor::<B, 1>::from(pi_target).to_device(device);
                let mask_vec : Vec<f32> = pi_target          // >0 exactly on legal moves
                    .iter().map(|&p| if p > 0.0 { 1.0 } else { 0.0 }).collect();
                let mask_t   = Tensor::<B,1>::from(mask_vec.as_slice()).to_device(device);

                let logp     = masked_log_softmax(pi_logits, mask_t.clone());
                let loss_p   = -(pi_t * logp).sum();

                let z_t   = Tensor::<B, 1>::from_floats([z], device);
                let loss_v = (v_pred - z_t.clone()).powf_scalar(2.0);
                let loss_r = (r_pred - z_t).powf_scalar(2.0);

                loss_t = loss_t + loss_p + loss_v + loss_r;
            }

            let loss  = loss_t / (batch_size as f32);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model     = opt.step(learning_rate.into(), model, grads);
        }
    }

    // final print if loop didn’t finish on a block boundary ---------------
    if games_count > 0 {
        println!("Mean Score : {:.3}", total_score / games_count as f32);
    }
    model
}

// ─────────────────────────────────────────────────────────────
//  Entry point: train then interactive test with *policy* head
// ─────────────────────────────────────────────────────────────
pub fn run_mu_zero<
    const N_S: usize,
    const N_A: usize,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Display + Default + Clone,
>(env_name: &str,p: DeepLearningParams) {
    let device = get_device();
    println!("Using device: {:?}", device);

    // out‑dim = policy (A) + value (1) + reward (1)
    let model = MyQmlp::<MyAutodiffBackend>::new(&device, N_S, N_A + 2);

    let name = format!("./data/mu_zero/{}", env_name);
    let mut logger = MuZeroLogger::new(&name, &p);
    let trained = episodic_mu_zero::<N_S, N_A, _, _, Env>(
        model,
        p.num_episodes,
        p.episode_stop,
        p.mz_replay_cap,
        p.mz_batch_size,
        p.mz_games_per_iter,
        p.mcts_simulations,
        p.ac_policy_lr,
        p.mz_c,
        p.opt_weight_decay_penalty,
        &device,
        &mut logger,
    );

    test_trained_model::<N_S, N_A, Env>(&device, trained);
}