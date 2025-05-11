use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;

pub const GRID_NUM_ACTIONS: usize = 4;
pub const NUM_COLS: usize = 5;
const NUM_BOARD_SIZE: usize = NUM_COLS*NUM_COLS;
pub const GRID_NUM_STATE_FEATURES: usize = NUM_BOARD_SIZE*2;

#[derive(Clone)]
pub struct GridWorld {
    pub board: [f32; NUM_BOARD_SIZE],
    pub player: u8,
    pub score: f32,
    pub is_game_over: bool,
    pub current_state: usize,
    pub is_random_state: bool,
    against_random: bool
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
            against_random: false
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

    fn available_actions_ids(&self) -> impl Iterator<Item=usize> {
        (0..GRID_NUM_ACTIONS).filter_map(|action| {
            let row = self.current_state / NUM_COLS;
            if (row == 0 && action == 0) || (row == NUM_COLS-1 && action == 1){
                None
            } else {
                Some(action)
            }
        })
    }

    fn available_actions(&self) -> impl Iterator<Item=usize> { self.available_actions_ids() }

    fn action_mask(&self) -> [f32; GRID_NUM_ACTIONS] {
        std::array::from_fn(|idx| {
            let row = self.current_state / NUM_COLS;
            if (row == 0 && idx == 0) || (row == NUM_COLS-1 && idx == 1) {
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
                { if row > 0 { row -= 1; } };
            }
            1 => { // Down
                { if row < NUM_COLS - 1   { row += 1; } };
            }
            2 => { // Left
                { if col > 0 { col -= 1; } };
            }
            3 => { // Right 
                { if col < NUM_COLS - 1   { col += 1; } };
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

    fn step_from_idx(&mut self, action_idx: usize) { self.step(action_idx) }

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

    fn set_from_random_state(&mut self) { self.is_random_state = !self.is_random_state }

    fn set_against_random(&mut self) -> bool {
        self.against_random = !self.against_random;
        self.against_random
    }

    fn state_index(&self) -> usize { self.current_state }

    fn switch_board(&mut self) {}
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
        writeln!(f,"Available actions: {:?}",self.available_actions().collect::<Vec<_>>())?;
        writeln!(f, "Game Over: {}", self.is_game_over)?;
        Ok(())
    }
}