mod algorithms;
mod environments;
mod services;
mod gui;
mod config;

use crate::services::envs::common::{clear_screen, end_of_run, user_choice};
use crate::services::envs::envs::{run_env_manually_random_1_v_1, run_env_manually_solo, run_deep_learning};
use crossterm::terminal::disable_raw_mode;
use environments::tic_tac_toe::{TicTacToeVersusRandom, TTT_NUM_ACTIONS, TTT_NUM_STATE_FEATURES};
use environments::line_world::{LineWorld, LINE_NUM_ACTIONS, LINE_NUM_STATE_FEATURES};
use environments::grid_world::{GridWorld, GRID_NUM_ACTIONS, GRID_NUM_STATE_FEATURES};


fn main() {
    let mut from_random = false;

    loop {
        let options = vec![
            if from_random { "Random is on, turn random off" } else { "Random is off, Turn random on" },
            "Line World - Heuristic",
            "Line World - DRL SARSA",
            "Grid World - Heuristic",
            "Grid World - DRL SARSA",
            "Tic Tac Toe - Heuristic",
            "Tic Tac Toe - DRL SARSA",
            "Tic Tac Toe - Benchmark",
            //"Bobail avec un autre joeur",
            //"Bobail avec un jouer random",
            //"Bobail benchmark test",
            "Quit"
        ];

        let selected_index = user_choice(options.clone());
        clear_screen();

        match selected_index {
            0 => { from_random = !from_random; }
            1 => { run_env_manually_solo::<LINE_NUM_STATE_FEATURES, LINE_NUM_ACTIONS, LineWorld>(&mut LineWorld::default(), options[selected_index]); },
            2 => { run_deep_learning::<LINE_NUM_STATE_FEATURES, LINE_NUM_ACTIONS, LineWorld>(); },
            3 => { run_env_manually_solo::<GRID_NUM_STATE_FEATURES, GRID_NUM_ACTIONS, GridWorld>(&mut GridWorld::default(), options[selected_index]); },
            4 => { run_deep_learning::<GRID_NUM_STATE_FEATURES, GRID_NUM_ACTIONS, GridWorld>(); },
            5 => { run_env_manually_solo::<TTT_NUM_STATE_FEATURES, TTT_NUM_ACTIONS, TicTacToeVersusRandom>(&mut TicTacToeVersusRandom::default(), options[selected_index]); },
            6 => { run_deep_learning::<TTT_NUM_STATE_FEATURES, TTT_NUM_ACTIONS, TicTacToeVersusRandom>(); },
            // 4 => { run_env_manually_solo(&mut environments::bobail::BobailEnv::new(), options[selected_index], from_random); },
            7 => { run_env_manually_random_1_v_1(&mut environments::bobail::BobailEnv::new(), options[selected_index], from_random); },
            // 8 => { benchmark_random_agents(&mut environments::bobail::BobailEnv::new(), options[selected_index], from_random); },
            8 => { break; }
            _ => {}
        }
        end_of_run();
    }
    disable_raw_mode().unwrap();
}
