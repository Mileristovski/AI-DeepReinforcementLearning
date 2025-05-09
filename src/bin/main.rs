mod algorithms;
mod environments;
mod services;
mod gui;
mod config;

use gui::cli::common::{clear_screen, end_of_run, user_choice};
use gui::cli::submenu::submenu;
use crossterm::terminal::disable_raw_mode;
use environments::tic_tac_toe::{TicTacToeVersusRandom, TTT_NUM_ACTIONS, TTT_NUM_STATE_FEATURES};
use environments::line_world::{LineWorld, LINE_NUM_ACTIONS, LINE_NUM_STATE_FEATURES};
use environments::grid_world::{GridWorld, GRID_NUM_ACTIONS, GRID_NUM_STATE_FEATURES};
use environments::bobail::{ BobailHeuristic, BB_NUM_ACTIONS, BB_NUM_STATE_FEATURES};


fn main() {
    loop {
        let options = vec![
            "Line World",
            "Grid World",
            "Tic Tac Toe",
            "Bobail",
            "Quit"
        ];

        let selected_index = user_choice(options.clone(), "Main Menu");
        clear_screen();

        match selected_index {
            0 => { submenu::<LINE_NUM_STATE_FEATURES, LINE_NUM_ACTIONS, LineWorld>(&mut LineWorld::default(), options[selected_index]); },
            1 => { submenu::<GRID_NUM_STATE_FEATURES, GRID_NUM_ACTIONS, GridWorld>(&mut GridWorld::default(), options[selected_index]); },
            2 => { submenu::<TTT_NUM_STATE_FEATURES, TTT_NUM_ACTIONS, TicTacToeVersusRandom>(&mut TicTacToeVersusRandom::default(), options[selected_index]); },
            3 => { submenu::<BB_NUM_STATE_FEATURES, BB_NUM_ACTIONS, BobailHeuristic>(&mut BobailHeuristic::default(), options[selected_index]); },
            4 => { break; }
            _ => {}
        }
        end_of_run();
    }
    disable_raw_mode().unwrap();
}
