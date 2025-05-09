mod algorithms;
mod environments;
mod services;
mod gui;
mod config;

use crate::services::envs::common::{clear_screen, end_of_run, user_choice};
use crate::services::envs::envs::{ run_env_heuristic, run_deep_learning, benchmark_random_agents};
use crossterm::terminal::disable_raw_mode;
use environments::tic_tac_toe::{TicTacToeVersusRandom, TTT_NUM_ACTIONS, TTT_NUM_STATE_FEATURES};
use environments::line_world::{LineWorld, LINE_NUM_ACTIONS, LINE_NUM_STATE_FEATURES};
use environments::grid_world::{GridWorld, GRID_NUM_ACTIONS, GRID_NUM_STATE_FEATURES};
use environments::bobail::{ BobailHeuristic, BB_NUM_ACTIONS, BB_NUM_STATE_FEATURES};


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
            "Bobail - Heuristic",
            "Bobail - Benchmark",
            "Quit"
        ];

        let selected_index = user_choice(options.clone());
        clear_screen();

        match selected_index {
            0 => { from_random = !from_random; }
            1 => { run_env_heuristic::<LINE_NUM_STATE_FEATURES, LINE_NUM_ACTIONS, LineWorld>(&mut LineWorld::default(), options[selected_index], from_random); },
            2 => { run_deep_learning::<LINE_NUM_STATE_FEATURES, LINE_NUM_ACTIONS, LineWorld>(); },
            3 => { run_env_heuristic::<GRID_NUM_STATE_FEATURES, GRID_NUM_ACTIONS, GridWorld>(&mut GridWorld::default(), options[selected_index], from_random); },
            4 => { run_deep_learning::<GRID_NUM_STATE_FEATURES, GRID_NUM_ACTIONS, GridWorld>(); },
            5 => { run_env_heuristic::<TTT_NUM_STATE_FEATURES, TTT_NUM_ACTIONS, TicTacToeVersusRandom>(&mut TicTacToeVersusRandom::default(), options[selected_index], from_random); },
            6 => { run_deep_learning::<TTT_NUM_STATE_FEATURES, TTT_NUM_ACTIONS, TicTacToeVersusRandom>(); },
            7 => { benchmark_random_agents::<TTT_NUM_STATE_FEATURES, TTT_NUM_ACTIONS, TicTacToeVersusRandom>(&mut TicTacToeVersusRandom::default(), options[selected_index]); },
            8 => { run_env_heuristic::<BB_NUM_STATE_FEATURES, BB_NUM_ACTIONS, BobailHeuristic>(&mut BobailHeuristic::default(), options[selected_index], from_random); },
            9 => { benchmark_random_agents::<BB_NUM_STATE_FEATURES, BB_NUM_ACTIONS, BobailHeuristic>(&mut BobailHeuristic::default(), options[selected_index]); },
            10 => { break; }
            _ => {}
        }
        end_of_run();
    }
    disable_raw_mode().unwrap();
}
