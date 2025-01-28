use nalgebra::{DVector};

pub trait Env {
    fn reset(state: &mut dyn State);
    fn display(state: &dyn State);
    fn is_game_over(state: &dyn State) -> bool;
    fn step(state: &mut dyn State, action: &dyn Action, reward: &dyn Reward);
}

pub trait State {
    fn num_states(&self) -> usize;
    fn state_id(&self) -> usize;
    fn update_state(&mut self, action: &dyn Action);
    fn start_from_random_state(&self);
    fn get_terminal_states(&self) -> Vec<usize>;
}

pub trait Action {
    fn available_actions(&self, state: &dyn State) -> DVector<i32>;
    fn num_actions(&self) -> usize;
    fn is_forbidden(&self, action: usize, state: &dyn State) -> bool;
}

pub trait Reward {
    fn num_rewards(&self) -> usize;
    fn get_reward_vector(&self) -> Vec<f32>;
    fn get_reward(&self, state: &dyn State) -> f32;
}

pub trait Score {
    fn get_score(&self) -> f32;
    fn update_score(&self, state: &mut dyn State, reward: &dyn Reward);
}