mod algorithms;
mod environments;
mod services;
mod gui;

use crossterm::terminal::disable_raw_mode;
use crate::services::testing::envs::testing_env_manually;
use crate::services::testing::common::{clear_screen, end_of_run, user_choice};

fn main() {
    let options = vec![
        "Line World",
        "Grid World",
        "Tic Tac Toe",
        "Bobeil",
        "Quit"
    ];

    let mut selected_index = 0;
    loop {
        selected_index = user_choice(options.clone());
        clear_screen();

        match selected_index {
            0 => { testing_env_manually(&mut environments::line_world::LineEnv::new(), options[selected_index]); },
            1 => { testing_env_manually(&mut environments::grid_world::GridEnv::new(), options[selected_index]); },
            2 => { testing_env_manually(&mut environments::tic_tac_toe::TicTacToeEnv::new(), options[selected_index]); },
            3 => { testing_env_manually(&mut environments::bobail::BobailEnv::new(), options[selected_index]); },
            4 => { break; }
            _ => {}
        }
        end_of_run();
    }
    disable_raw_mode().unwrap();
}
