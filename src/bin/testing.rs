mod services;
mod environments;

use crossterm::terminal::disable_raw_mode;
use crate::services::testing::envs::testing_env_manually;
use crate::services::testing::common::{clear_screen, end_of_run, user_choice};

pub fn main() {
    let options = vec![
        "Line World",
        "Grid World",
        "Tic Tac Toe",
        "Quit"
    ];

    let mut selected_index = 0;
    loop {
        selected_index = user_choice(options.clone());
        clear_screen();

        match selected_index {
            0 => { testing_env_manually(&mut environments::line_world::LineEnv::new()); },
            1 => { testing_env_manually(&mut environments::grid_world::GridEnv::new()); },
            2 => { testing_env_manually(&mut environments::tic_tac_toe::TicTacToeEnv::new()); },
            3 => { break; }
            _ => {}
        }
        end_of_run();
    }
    disable_raw_mode().unwrap();
}
