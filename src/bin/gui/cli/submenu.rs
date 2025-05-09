use std::fmt::Display;
use std::io;
use crate::gui::cli::common::{reset_screen, user_choice};
use crate::services::envs::run::{run_env_heuristic, run_benchmark_random_agents};
use crate::algorithms::sarsa::run_episodic_semi_gradient_sarsa;
use crossterm::terminal::disable_raw_mode;
use crate::environments::env::DeepDiscreteActionsEnv;

pub fn submenu<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(env: &mut Env, env_name: &str) {
    let mut from_random = false;
    
    loop {
        let options = vec![
            if from_random { "Random is on, turn random off" } else { "Random is off, Turn random on" },
            "Heuristic",
            "Train a model from the env",
            "Benchmark",
            "Quit"
        ];
        
        let message = format!("Menu for {}", env_name);
        let selected_index = user_choice(options.clone(), &message);
        let mut stdout = io::stdout();
        reset_screen(&mut stdout, env_name);

        match selected_index {
            0 => { from_random = !from_random; }
            1 => { run_env_heuristic::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env, options[selected_index], from_random); },
            2 => { submenu_drl::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name); },
            3 => { run_benchmark_random_agents::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env, options[selected_index], from_random); },
            4 => { break; }
            _ => {}
        }
    }
    disable_raw_mode().unwrap();
}

pub fn submenu_drl<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(env_name: &str) {
    loop {
        let options = vec![
            "Semi gradient SARSA",
            "Quit"
        ];
        
        let message = format!("Training menu for {}", env_name);
        let selected_index = user_choice(options.clone(), &message);
        let mut stdout = io::stdout();
        reset_screen(&mut stdout, env_name);

        match selected_index {
            0 => { run_episodic_semi_gradient_sarsa::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(); },
            1 => { break; }
            _ => {}
        }
    }
    disable_raw_mode().unwrap();
}

