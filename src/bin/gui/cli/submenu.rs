use std::fmt::Display;
use std::io;
use crate::gui::cli::common::{end_of_run, reset_screen, user_choice};
use crate::services::envs::run::{run_env_heuristic, run_deep_learning, benchmark_random_agents};
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
            "Benchmark",
            "DRL SARSA",
            "Quit"
        ];

        let selected_index = user_choice(options.clone(), env_name);
        let mut stdout = io::stdout();
        reset_screen(&mut stdout, env_name);

        match selected_index {
            0 => { from_random = !from_random; }
            1 => { run_env_heuristic::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env, options[selected_index], from_random); },
            2 => { run_deep_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(); },
            3 => { benchmark_random_agents::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env, options[selected_index]); },
            4 => { break; }
            _ => {}
        }
        end_of_run();
    }
    disable_raw_mode().unwrap();
}