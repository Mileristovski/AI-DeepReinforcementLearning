use std::fmt::Display;
use std::io;
use crate::gui::cli::common::{reset_screen, user_choice};
use crate::services::envs::run::{run_env_heuristic, run_benchmark_random_agents};
use crate::algorithms::sarsa::run_episodic_semi_gradient_sarsa;
use crossterm::terminal::disable_raw_mode;
use crate::algorithms::q_learning::deep_q_learning::run_deep_q_learning;
use crate::algorithms::q_learning::double_deep_q_learning::run_double_deep_q_learning;
use crate::algorithms::q_learning::double_deep_q_learning_with_experience_replay::run_double_deep_q_learning_er;
use crate::algorithms::q_learning::double_deep_q_learning_with_prioritized_experience_replay::run_double_dqn_per;
use crate::algorithms::q_learning::tabular_q_learning::run_tabular_q_learning;
use crate::algorithms::reinforce::reinforce::run_reinforce;
use crate::environments::env::DeepDiscreteActionsEnv;

pub fn submenu<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(env: &mut Env, env_name: &str) {
    let mut from_random = false;
    
    loop {
        let options = vec![
            if from_random { "Random is on, turn random off" } else { "Random is off, Turn random on" },
            "Heuristic",
            "Train a model from the env",
            "Benchmark",
            "Back"
        ];
        
        let message = format!("Menu for {}", env_name);
        let selected_index = user_choice(options.clone(), &message);
        let mut stdout = io::stdout();
        reset_screen(&mut stdout, env_name);

        match selected_index {
            0 => { from_random = !from_random; }
            1 => { run_env_heuristic::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env, options[selected_index], from_random); },
            2 => { submenu_drl::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name); },
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
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(env_name: &str) {
    loop {
        let options = vec![
            "Semi gradient SARSA",
            "Tabular Q Learning",
            "Deep Q Learning",
            "Double Deep Q Learning",
            "Double Deep Q Learning With Experienced Replay",
            "Double Deep Q Learning With Prioritized Experienced Replay",
            "REINFORCE",
            "Back"
        ];
        
        let message = format!("Training menu for {}", env_name);
        let selected_index = user_choice(options.clone(), &message);
        let mut stdout = io::stdout();
        reset_screen(&mut stdout, env_name);

        match selected_index {
            0 => { run_episodic_semi_gradient_sarsa::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(); },
            1 => { run_tabular_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(); },
            2 => { run_deep_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(); },
            3 => { run_double_deep_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(); },
            4 => { run_double_deep_q_learning_er::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(); },
            5 => { run_double_dqn_per::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(); },
            6 => { run_reinforce::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(); },
            7 => { break; }
            _ => {}
        }
    }
    disable_raw_mode().unwrap();
}

