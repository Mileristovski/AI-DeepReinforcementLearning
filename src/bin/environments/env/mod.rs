    use nalgebra::{DVector};

    pub trait Env {
        fn num_states(&self) -> usize;
        fn num_actions(&self) -> usize;
        fn num_rewards(&self) -> usize;
        fn get_reward_vector(&self) -> Vec<f32>;
        fn get_terminal_states(&self) -> Vec<usize>;
        fn get_reward(&self, num: usize) -> f32;
        fn state_id(&self) -> Vec<i32>;
        fn reset(&mut self);
        fn display(&self);
        fn is_forbidden(&self, action: i32) -> bool;
        fn is_game_over(&self) -> bool;
        fn available_actions(&self) -> DVector<i32>;
        fn step(&mut self, action: i32);
        fn score(&self) -> f32;
        fn start_from_random_state(&mut self);
        fn transition_probability(&self, s: usize, a: usize, s_p: usize, r_index: usize) -> f32;
    }
