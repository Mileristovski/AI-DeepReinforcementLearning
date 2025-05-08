mod services;
mod environments;
mod algorithms;
mod config;

use crate::environments::env::DeepDiscreteActionsEnv;
use crate::environments::grid_world::GridWorld;

type GameEnv = GridWorld;

fn display_properties(env: &mut GameEnv) {
    println!("{:?}", env.state_description());
    println!("{:?}", env.board);
    println!("{:?}", env.available_actions_ids().collect::<Vec<_>>());
    println!("{:?}", env.action_mask());
}

fn main() {
    let mut env = GameEnv::default();
    // env.is_random_state = true;
    env.reset();
    
    println!("{}", env);
    display_properties(&mut env);
    env.step(1);
    display_properties(&mut env);
    println!("{}", env);
    
    env.step(0);
    display_properties(&mut env);
    println!("{}", env);
        
    env.step(3);
    display_properties(&mut env);
    println!("{}", env);

    env.step(3);
    display_properties(&mut env);
    println!("{}", env);
    
    display_properties(&mut env);
    unimplemented!("This should be used as a test script")
}
