use burn::backend::{Autodiff, LibTorch};
use burn::module::AutodiffModule;
use burn::prelude::Backend;
/**
* -------------------------------------------------------------------------
* BACKEND AND DEVICE INITIALIZATION
* -------------------------------------------------------------------------
*/
pub type MyBackend = LibTorch;
pub type MyAutodiffBackend = Autodiff<MyBackend>;
pub type MyDevice = <MyBackend as Backend>::Device;
pub type Enemy<M, B> = <M as AutodiffModule<B>>::InnerModule;
pub const EXPORT_AT_EP: [usize; 4] = [1000, 2500, 100_000, 1_000_000];
pub const REPLAY_CAPACITY: usize = 100_000;
pub const BATCH_SIZE: usize = 32;
pub const TARGET_UPDATE_EVERY: usize = 1_000;
pub const CAPACITY: usize = 100_000;
pub const BATCH:    usize = 64;
pub const TARGET_EVERY: usize = 10_000;          // gradient steps
pub const PRIO_EPS:  f32 = 1e-6;
pub const PRIO_MAX:  f32 = 10.0;
pub const BETA_START:f32 = 0.4;
pub const BETA_END:  f32 = 1.0;
pub const BETA_FRAMES: usize = 1_000_000;
/**
* -------------------------------------------------------------------------
* NEURAL NETWORK HIDDEN LAYERS
* -------------------------------------------------------------------------
*/
pub fn hidden_sizes() -> Vec<usize> {
    vec![512, 256, 256, 128]
}

/**
* -------------------------------------------------------------------------
* DEEP LEARNING ALGORITHMS
* -------------------------------------------------------------------------
*/
pub struct DeepLearningParams {
    // … your existing fields …
    pub num_episodes: usize,
    pub episode_stop: usize,
    pub start_epsilon: f32,
    pub final_epsilon: f32,

    // Q‑learning / SARSA
    pub gamma: f32,
    pub alpha: f32,        // base LR
    pub ql_per_alpha: f32,
    pub ql_replay_capacity: usize,
    pub ql_batch_size: usize,
    pub ql_target_update_every: usize,
    
    
    // actor‑critic / A2C extras
    pub ac_n_step: usize,
    pub ac_entropy_coef: f32,
    pub ac_policy_lr: f32,
    pub ac_critic_lr: f32,

    // MCTS / AlphaZero
    pub mcts_rollouts_per_action: usize,
    pub mcts_simulations: usize,
    pub mcts_c: f32,

    // AlphaZero‑specific    
    pub az_self_play_games: usize,     
    pub az_c: f32,
    
    // MuZero-specific
    pub mz_games_per_iter: usize,
    pub mz_replay_cap: usize,
    pub mz_batch_size: usize,
    pub mz_c: f32,
    pub opt_weight_decay_penalty: f32,

    // RNG
    pub rng_seed: u64,
}

impl Default for DeepLearningParams {
    fn default() -> Self {
        Self {
            num_episodes: 10000,
            episode_stop: 1000,
            start_epsilon: 1.0,
            final_epsilon: 1e-5,
            
            // Q‑learning / SARSA
            gamma: 0.999,
            alpha: 3e-3,
            ql_per_alpha: 0.6,
            ql_replay_capacity: 100_000,
            ql_batch_size: 32,
            ql_target_update_every: 1000,
            

            // actor‑critic
            ac_n_step: 5,
            ac_entropy_coef: 0.01,
            ac_policy_lr: 3e-3,
            ac_critic_lr: 6e-3,

            // MCTS
            mcts_rollouts_per_action: 50,
            mcts_simulations: 100,
            mcts_c: 1.414,

            // AlphaZero
            az_self_play_games: 1,
            az_c: 1.4,

            // MuZero
            mz_games_per_iter: 1,
            mz_replay_cap: 100_000,
            mz_batch_size: 256,
            mz_c: 1.0,
            
            // Optimiser settings Adam
            opt_weight_decay_penalty: 1e-4,
            
            // RNG
            rng_seed: 172_848_556_3,
        }
    }
}
