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
    vec![64, 32, 16, 16]
}

/**
* -------------------------------------------------------------------------
* DEEP LEARNING ALGORITHMS
* -------------------------------------------------------------------------
*/
pub struct DeepLearningParams {
    pub num_episodes: usize,
    pub gamma: f32,
    pub alpha: f32,
    pub start_epsilon: f32,
    pub final_epsilon: f32
}

// Parameters for Deep Learning Algorithms 
impl Default for DeepLearningParams {
    fn default() -> Self {
        Self {
            num_episodes: 5_000,
            gamma:0.999f32,
            alpha: 3e-3,
            start_epsilon: 1.0f32,
            final_epsilon: 1e-5f32
        }
    }
}
