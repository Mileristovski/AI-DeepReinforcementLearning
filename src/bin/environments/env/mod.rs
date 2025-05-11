pub trait DeepDiscreteActionsEnv<const NUM_STATES_FEATURES: usize, const NUM_ACTIONS: usize>: Default + Clone {
    fn state_description(&self) -> [f32; NUM_STATES_FEATURES];
    fn available_actions_ids(&self) -> impl Iterator<Item=usize>;
    fn available_actions(&self) -> impl Iterator<Item=usize>;
    fn action_mask(&self) -> [f32; NUM_ACTIONS];
    fn step(&mut self, action: usize);
    fn is_game_over(&self) -> bool;
    fn score(&self) -> f32;
    fn reset(&mut self);
    fn set_from_random_state(&mut self);
    fn set_against_random(&mut self) -> bool;
    fn step_from_idx(&mut self, action_idx: usize);
    fn state_index(&self) -> usize;
    fn switch_board(&mut self); 
}
