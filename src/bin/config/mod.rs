use burn::backend::{Autodiff, LibTorch};
use burn::prelude::Backend;

/**
* -------------------------------------------------------------------------
* BACKEND AND DEVICE INITIALIZATION
* -------------------------------------------------------------------------
*/
pub type MyBackend = LibTorch;
pub type MyAutodiffBackend = Autodiff<MyBackend>;
pub type MyDevice = <MyBackend as Backend>::Device;


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
    pub per_alpha: f32,

    // actor‑critic / A2C extras
    pub n_step: usize,
    pub entropy_coef: f32,
    pub policy_lr: f32,
    pub critic_lr: f32,

    // MCTS / AlphaZero
    pub rollouts_per_action: usize,
    pub mcts_simulations: usize,
    pub mcts_c: f32,

    // AlphaZero‑specific
    pub az_iterations: usize,         // outer training loop (“700k” in paper)
    pub az_self_play_games: usize,    // self‑play games per iteration
    // pub az_batch_size: usize,         // how many positions to batch‐train on
    // pub az_weight_decay: f32,         // L2 reg on θ
    pub az_apprentice_prob: f32,
    pub c: f32,
    
    // MuZero-specific
    pub mz_games_per_iter: usize,
    pub mz_replay_cap: usize,
    pub mz_batch_size: usize,
    pub mz_c: f32,
    pub mz_weight_decay_penalty: f32,

    // RNG
    pub rng_seed: u64,
}

impl Default for DeepLearningParams {
    fn default() -> Self {
        Self {
            num_episodes: 10_000,
            episode_stop: 1_000,
            start_epsilon: 1.0,
            final_epsilon: 1e-5,
            
            // Q‑learning / SARSA
            gamma: 0.999,
            alpha: 3e-3,
            per_alpha: 0.6,

            // actor‑critic
            n_step: 5,
            entropy_coef: 0.01,
            policy_lr: 3e-3,
            critic_lr: 6e-3,

            // MCTS
            rollouts_per_action: 50,
            mcts_simulations: 100,
            mcts_c: 1.414,

            // AlphaZero
            az_iterations: 700_000,
            az_self_play_games: 1,
            // az_batch_size: 256,
            // az_weight_decay: 1e-4,
            az_apprentice_prob: 0.5,
            c: 1.4,

            // MuZero
            mz_games_per_iter: 1,
            mz_replay_cap: 100_000,
            mz_batch_size: 256,
            mz_c: 1.0,
            mz_weight_decay_penalty: 1e-4,
            
            // RNG
            rng_seed: 172_848_556_3,
        }
    }
}
