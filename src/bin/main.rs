mod algorithms;
mod environments;
mod services;
mod gui;
mod config;

use crate::services::envs::envs::benchmark_random_agents;
use crate::services::envs::common::{clear_screen, end_of_run, user_choice};
use crate::services::envs::envs::{run_env_manually_random_1_v_1, run_env_manually_solo, run_deep_learning};
use crossterm::terminal::disable_raw_mode;
use environments::tic_tac_toe::tic_tac_toe::{TicTacToeVersusRandom, NUM_ACTIONS, NUM_STATE_FEATURES};

fn main() {
    let mut from_random = false;

    loop {
        let options = vec![
            if from_random { "Random is on, turn random off" } else { "Random is off, Turn random on" },
            "Line World",
            "Grid World",
            "Tic Tac Toe",
            "Bobail avec un autre joeur",
            "Bobail avec un jouer random",
            "Bobail benchmark test",
            "Quit"
        ];

        let selected_index = user_choice(options.clone());
        clear_screen();

        match selected_index {
            0 => { from_random = !from_random; }
            1 => { run_env_manually_solo(&mut environments::line_world::LineEnv::new(), options[selected_index], from_random); },
            2 => { run_env_manually_solo(&mut environments::grid_world::GridEnv::new(), options[selected_index], from_random); },
            3 => { run_deep_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, TicTacToeVersusRandom>(); },
            4 => { run_env_manually_solo(&mut environments::bobail::BobailEnv::new(), options[selected_index], from_random); },
            5 => { run_env_manually_random_1_v_1(&mut environments::bobail::BobailEnv::new(), options[selected_index], from_random); },
            6 => { benchmark_random_agents(&mut environments::bobail::BobailEnv::new(), options[selected_index], from_random); },
            7 => { break; }
            _ => {}
        }
        end_of_run();
    }
    disable_raw_mode().unwrap();
}
