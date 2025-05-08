use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;

pub const GRID_NUM_ACTIONS: usize = 4;
pub const NUM_COLS: usize = 5;
pub const NUM_BOARD_SIZE: usize = NUM_COLS*NUM_COLS;
pub const GRID_NUM_STATE_FEATURES: usize = NUM_BOARD_SIZE*2;

#[derive(Clone)]
pub struct GridWorld {
    pub board: [f32; NUM_BOARD_SIZE],
    pub player: u8,
    pub score: f32,
    pub is_game_over: bool,
    pub current_state: usize,
    pub is_random_state: bool,
}

impl Default for GridWorld {
    fn default() -> Self {
        let mut board = [0f32; NUM_BOARD_SIZE];
        board[NUM_BOARD_SIZE/2] = 1.0;
        Self {
            board,
            player: 0,
            score: 0.0,
            is_game_over: false,
            current_state: NUM_BOARD_SIZE/2,
            is_random_state: false,
        }
    }
}

impl DeepDiscreteActionsEnv<GRID_NUM_STATE_FEATURES, GRID_NUM_ACTIONS> for GridWorld {
    fn state_description(&self) -> [f32; GRID_NUM_STATE_FEATURES] {
        std::array::from_fn(|idx| {
            let cell = idx / 2;
            let feature = idx % 2;
            if self.board[cell] == feature as f32 {
                1.0
            } else {
                0.0
            }
        })
    }

    fn available_actions_ids(&self) -> impl Iterator<Item=usize> { [0, 1, 2, 3].into_iter() }

    fn action_mask(&self) -> [f32; GRID_NUM_ACTIONS] {
        std::array::from_fn(|idx| {
            let row = self.current_state % NUM_COLS;
            if row == 0 && idx == 0{
                0.0
            } else if row == NUM_COLS-1 && idx == 1 {
                0.0
            } else {
                1.0
            }
        })
    }

    fn step(&mut self, action: usize) {
        if self.is_game_over {
            panic!("Trying to play while Game is Over");
        }

        if action >= GRID_NUM_ACTIONS {
            panic!("Invalid action : {}", action);
        }

        let mut row = self.current_state / NUM_COLS;
        let mut col = self.current_state % NUM_COLS;
        // Update board
        match action {
            0 => { // Up
                row += 1;
            }
            1 => { // Down
                row -= 1;
            }
            2 => { // Left
                col += 1;
            }
            3 => { // Right 
                col -= 1;
            }
            _ => panic!("Invalid action")
        }
        // Return to correct 1D Vector
        self.board[self.current_state] = 0.0;
        self.current_state = row * NUM_COLS + col;
        self.board[self.current_state] = 1.0;

        // Check if game is over
        if col == NUM_COLS-1 {
            self.is_game_over = true;
            self.score = 1.0;
        } else if col == 0 {
            self.is_game_over = true;
            self.score = -1.0;
        }
    }

    fn is_game_over(&self) -> bool { self.is_game_over }

    fn score(&self) -> f32 { self.score }

    fn reset(&mut self) {
        self.board = [0f32; NUM_BOARD_SIZE];
        self.board[NUM_BOARD_SIZE/2] = 1.0;
        self.player = 0;
        self.score = 0.0;
        self.is_game_over = false;
        self.current_state = NUM_BOARD_SIZE/2;
        self.is_random_state = false;
    }
}

impl Display for GridWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let agent_row = self.current_state / NUM_COLS;
        let agent_col = self.current_state % NUM_COLS;
        for row in 0..NUM_COLS {
            for col in 0..NUM_COLS {
                if row == agent_row && col == agent_col {
                    f.write_str("X ")?;
                } else {
                    f.write_str("_ ")?;
                }
            }
            f.write_str("\n")?;
        }
        f.write_str("\n")?;
        writeln!(f, "Score: {}", self.score)?;
        writeln!(f, "Game Over: {}", self.is_game_over)?;
        Ok(())
    }
}