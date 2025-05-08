use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use rand::Rng;

pub const LINE_NUM_ACTIONS: usize = 2;
pub const NUM_BOARD_SIZE: usize = 5;
pub const LINE_NUM_STATE_FEATURES: usize = NUM_BOARD_SIZE*2;

#[derive(Clone)]
pub struct LineWorld {
    pub board: [f32; NUM_BOARD_SIZE],
    pub score: f32,
    pub is_game_over: bool,
    pub current_state: usize,
    pub is_random_state: bool,
}

impl Default for LineWorld {
    fn default() -> Self {
        let mut board = [0f32; NUM_BOARD_SIZE];
        board[NUM_BOARD_SIZE/2] = 1.0;

        Self {
            board,
            score: 0.0,
            is_game_over: false,
            current_state: NUM_BOARD_SIZE/2,
            is_random_state: false,
        }
    }
}

impl DeepDiscreteActionsEnv<LINE_NUM_STATE_FEATURES, LINE_NUM_ACTIONS> for LineWorld {
    fn state_description(&self) -> [f32; LINE_NUM_STATE_FEATURES] {
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

    fn available_actions_ids(&self) -> impl Iterator<Item=usize> { [0, 1].into_iter() }

    fn action_mask(&self) -> [f32; LINE_NUM_ACTIONS] { [1.0f32; LINE_NUM_ACTIONS] }

    fn step(&mut self, action: usize) {
        if self.is_game_over {
            panic!("Trying to play while Game is Over");
        }

        if action >= LINE_NUM_ACTIONS {
            panic!("Invalid action : {}", action);
        }

        // Update board
        self.board[self.current_state] = 0.0;
        if action == 0 {
            self.board[self.current_state-1] = 1.0;
            self.current_state -= 1;
        } else {
            self.board[self.current_state+1] = 1.0;
            self.current_state += 1;
        }

        // Check if game is over
        if self.current_state == 0 {
            self.score = -1.0;
            self.is_game_over = true;
        } else if self.current_state == NUM_BOARD_SIZE-1 {
            self.score = 1.0;
            self.is_game_over = true;
        }
    }

    fn is_game_over(&self) -> bool { self.is_game_over }

    fn score(&self) -> f32 { self.score }

    fn reset(&mut self) {
        self.board = [0f32; NUM_BOARD_SIZE];
        self.board[NUM_BOARD_SIZE/2] = 1.0;
        self.score =  0.0;
        self.is_game_over =  false;
        self.current_state = if self.is_random_state { 
            rand::thread_rng().gen_range(1..NUM_BOARD_SIZE-1) 
        } else { 
            NUM_BOARD_SIZE / 2 
        };
    }
}

impl Display for LineWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for s in 0..NUM_BOARD_SIZE {
            if s == self.current_state {
                f.write_str("X")?;
            } else {
                f.write_str("_")?;
            }
        }
        f.write_str("\n")?;
        writeln!(f, "Score: {}", self.score)?;
        writeln!(f, "Game Over: {}", self.is_game_over)?;
        Ok(())
    }
}
