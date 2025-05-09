use crate::environments::env::DeepDiscreteActionsEnv;
use rand::prelude::IteratorRandom;
use std::fmt::Display;
use rand::thread_rng;
use std::collections::HashMap;

pub const BB_NUM_STATE_FEATURES: usize = 300;
pub const BB_NUM_ACTIONS: usize = 48;
const ROWS: usize = 5;
const COLS: usize = 5;
const NUM_BOARD_SIZE: usize = ROWS*COLS;

#[derive(Clone)]
pub struct BobailHeuristic {
    pub board: [f32; NUM_BOARD_SIZE],
    pub current_player: usize,
    pub previous_player: usize,
    is_game_over: bool,
    winner: usize,
    terminal_states: Vec<usize>,
    score: f32,
    r: Vec<f32>,
    rows: usize,
    cols: usize,
    pub blue_player: usize,
    pub red_player: usize,
    pub bobail: usize,
    bobail_move: bool,
    directions:  [(isize, isize, isize); 8],
    empty: f32,
    chip_lookup: HashMap<usize, usize>,
    pub is_random_state: bool,
    against_random: bool
}

impl BobailHeuristic {
    pub fn new() -> Self {
        let chip_lookup: HashMap<usize, usize> = HashMap::from([
            // blue
            (6,  0), (7,  1), (8,  2), (9,  3), (10,  4),
            // red
            (1, 20), (2, 21), (3, 22), (4, 23), (5, 24),
            // bobail
            (11, NUM_BOARD_SIZE / 2),
        ]);

        let mut board = [0.0f32; NUM_BOARD_SIZE];
        for red in 0..COLS {
            board[red] = red as f32 + 6.0;
        }
        // Add the red player
        for blue in NUM_BOARD_SIZE-COLS..NUM_BOARD_SIZE {
            board[blue] = blue as f32 - ((NUM_BOARD_SIZE-COLS) as f32 - 1.0);
        }
        // Add the bobail
        board[NUM_BOARD_SIZE/2] = 11.0;
        Self {
            current_player: 1,
            previous_player: 1,
            board,
            is_game_over: false,
            winner: 0,
            terminal_states: vec!(0usize, 1, 2, 3, 4, 20, 21, 22, 23, 24),
            score: 0.0,
            r: vec![-1.0f32, 0.0, 1.0],
            rows: ROWS,
            cols: COLS,
            blue_player: 1,
            red_player: 2,
            bobail: 3,
            bobail_move: false,
            directions: [
                (-1, 0, 0),     // Up
                (1, 0, 1),      // Down
                (0, -1, 2),     // Left
                (0, 1, 3),      // Right
                (-1, -1, 4),    // Up-Left
                (-1, 1, 5),     // Up-Right
                (1, -1, 6),     // Down-Left
                (1, 1, 7),      // Down-Right
            ],
            empty: 0.0,
            chip_lookup,
            is_random_state: false,
            against_random: false,
        }
    }

    fn decode_action(&self, action: usize) -> (usize, usize, usize, usize) {
        let chip_id   = action / 10;       // 1-11
        let direction = action % 10;       // 0-7

        let index = *self.chip_lookup
            .get(&chip_id)
            .expect("chip_id missing from lookup");

        let row = index / self.cols;
        let col = index % self.cols;

        (chip_id, row, col, direction)
    }

    fn get_possible_moves(&self, row: usize, col: usize) -> Vec<usize> {
        let mut possible_moves = Vec::new();

        for &(dr, dc, direction) in &self.directions {
            let new_row = row as isize + dr;
            let new_col = col as isize + dc;

            // Check if the new position is within bounds
            if new_row >= 0 && new_row < self.rows as isize && new_col >= 0 && new_col < self.cols as isize {
                let index = (new_row as usize) * self.cols + (new_col as usize);
                if self.board[index] == self.empty { // Only move if the spot is empty
                    possible_moves.push(direction as usize);
                }
            }
        }
        possible_moves
    }

    fn move_piece(&mut self, action: usize) -> (usize, usize) {
        let (chip_id, row, col, dir) = self.decode_action(action);

        let (dr, dc, _) = self
            .directions
            .iter()
            .copied()
            .find(|&(_, _, d)| d == dir as isize)
            .expect("Invalid direction");

        let mut new_row = row as isize;
        let mut new_col = col as isize;

        if self.bobail_move {
            new_row += dr;
            new_col += dc;
        } else {
            loop {
                let next_row = new_row + dr;
                let next_col = new_col + dc;

                if next_row < 0
                    || next_row >= self.rows as isize
                    || next_col < 0
                    || next_col >= self.cols as isize
                {
                    break;
                }

                let index = (next_row as usize) * self.cols + next_col as usize;
                if self.board[index] != self.empty {
                    break;
                }

                new_row = next_row;
                new_col = next_col;
            }
        }

        let dst_index = new_row as usize * self.cols + new_col as usize;
        let src_index = row * self.cols + col;

        self.board[src_index] = self.empty;
        self.board[dst_index] = chip_id as f32;
        self.chip_lookup.insert(chip_id, dst_index);

        (new_row as usize, new_col as usize)
    }

    fn display_helper(num: usize) -> String {
        match num {
            1..=10 => format!("{:^4}", if num <= 5 {
                format!("B{}", num)
            } else {
                format!("R{}", num)
            }),
            11 => format!("{:^4}", "Y11"),
            _ => "    ".to_string(),
        }
    }
}

impl Default for BobailHeuristic {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepDiscreteActionsEnv<BB_NUM_STATE_FEATURES, BB_NUM_ACTIONS> for BobailHeuristic {
    fn state_description(&self) -> [f32; BB_NUM_STATE_FEATURES] {
        std::array::from_fn(|idx| {
            let cell = idx / 12;
            let feature = idx % 12;
            if self.board[cell] == feature as f32 {
                1.0
            } else {
                0.0
            }
        })
    }

    fn available_actions_ids(&self) -> impl Iterator<Item = usize> {
        let mut available_actions = Vec::new();

        // Choose which chips belong to the current player
        let controlled_chip_ids: Vec<usize> = if self.bobail_move {
            vec![11]
        } else {
            match self.current_player {
                1 => vec![1, 2, 3, 4, 5],       // Blue player
                2 => vec![6, 7, 8, 9, 10],      // Red player
                _ => vec![]
            }
        };


        for &chip_id in &controlled_chip_ids {
            if let Some(&index) = self.chip_lookup.get(&chip_id) {
                let row = index / self.cols;
                let col = index % self.cols;

                let directions = self.get_possible_moves(row, col);

                for &dir in &directions {
                    // Action = 10 * chip_id + direction
                    let action = chip_id * 10 + dir;
                    available_actions.push(action);
                }
            }
        }

        available_actions.into_iter()
    }

    fn action_mask(&self) -> [f32; BB_NUM_ACTIONS] {
        let mut mask = [0.0; BB_NUM_ACTIONS];

        // Only generate when it's Blue's turn
        if self.current_player != self.blue_player {
            return mask;
        }

        // Index 0 is bobail, 1-5 are blue chips
        let agent_chip_ids = [11, 1, 2, 3, 4, 5];

        for (i, &chip_id) in agent_chip_ids.iter().enumerate() {
            if let Some(&index) = self.chip_lookup.get(&chip_id) {
                if (self.bobail_move && [11].contains(&chip_id)) || (!self.bobail_move && [1, 2, 3, 4, 5].contains(&chip_id)) {
                    let row = index / self.cols;
                    let col = index % self.cols;

                    for dir in self.get_possible_moves(row, col) {
                        let mask_index = i * 8 + dir;
                        mask[mask_index] = 1.0;
                    }
                }
            }
        }

        mask
    }

    fn step(&mut self, action: usize) {
        if self.is_game_over {
            panic!("Trying to play while Game is Over");
        }

        // Move the piece
        let (new_row, _) = Self::move_piece(self, action);

        self.bobail_move = !self.bobail_move;

        // Verify if the game is over
        self.is_game_over = {
            let bobail_index = self.chip_lookup.get(&11);

            let bobail_in_terminal = match bobail_index {
                Some(&idx) => self.terminal_states.contains(&idx),
                None => false,
            };

            let no_actions_left = self.available_actions_ids().next().is_none();

            bobail_in_terminal || no_actions_left
        };

        if self.is_game_over {
            self.winner = if new_row == 0 {
                self.red_player.clone()
            } else if new_row == self.rows-1 {
                self.blue_player.clone()
            } else {
                self.previous_player.clone()
            };

            // Add the rewards to the score
            if self.current_player == self.blue_player {
                self.score += 1.0;
            } else  {
                self.score -= 1.0;
            }
        };

        if self.bobail_move {
            self.current_player = if self.current_player == self.blue_player {
                self.red_player
            } else {
                self.blue_player
            };
        }

        if self.against_random && self.current_player == self.red_player {
            // random move
            let mut rng = thread_rng();
            let random_action = self.available_actions_ids().choose(&mut rng).unwrap();
            self.step(random_action);
        }
    }

    fn is_game_over(&self) -> bool { self.is_game_over }

    fn score(&self) -> f32 { self.score }

    fn reset(&mut self) {
        let chip_lookup: HashMap<usize, usize> = HashMap::from([
            // blue
            (6,  0), (7,  1), (8,  2), (9,  3), (10,  4),
            // red
            (1, 20), (2, 21), (3, 22), (4, 23), (5, 24),
            // bobail
            (11, NUM_BOARD_SIZE / 2),
        ]);

        self.board = [0.0f32; NUM_BOARD_SIZE];
        for red in 0..COLS {
            self.board[red] = red as f32 + 6.0;
        }
        // Add the red player
        for blue in NUM_BOARD_SIZE-COLS..NUM_BOARD_SIZE {
            self.board[blue] = blue as f32 - ((NUM_BOARD_SIZE-COLS) as f32 - 1.0);
        }
        // Add the bobail
        self.board[NUM_BOARD_SIZE/2] = 11.0;

        self.current_player = 1;
        self.previous_player = 1;
        self.is_game_over = false;
        self.winner = 0;
        self.terminal_states = vec!(0usize, 1, 2, 3, 4, 20, 21, 22, 23, 24);
        self.score = 0.0;
        self.r = vec![-1.0f32, 0.0, 1.0];
        self.rows = ROWS;
        self.cols = COLS;
        self.blue_player = 1;
        self.red_player = 2;
        self.bobail = 3;
        self.bobail_move = false;
        self.directions=  [
            (-1, 0, 0),     // Up
            (1, 0, 1),      // Down
            (0, -1, 2),     // Left
            (0, 1, 3),      // Right
            (-1, -1, 4),    // Up-Left
            (-1, 1, 5),     // Up-Right
            (1, -1, 6),     // Down-Left
            (1, 1, 7),      // Down-Right
        ];
        self.empty = 0.0
    }

    fn set_from_random_state(&mut self) { self.is_random_state = !self.is_random_state }

    fn set_against_random(&mut self) { self.against_random = !self.against_random }
}

impl Display for BobailHeuristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, " {}", "----+".repeat(self.cols - 1) + "----")?;

        for row in 0..self.rows {
            write!(f, "|")?;
            for col in 0..self.cols {
                let idx = row * self.cols + col;
                let val = self.board[idx] as usize;
                let cell = Self::display_helper(val);
                write!(f, "{}|", cell)?;
            }
            writeln!(f)?;
            writeln!(f, " {}", "----+".repeat(self.cols - 1) + "----")?;
        }

        writeln!(f)?;
        writeln!(f, "Score: {}", self.score)?;
        writeln!(f, "Player to move: {}", match self.current_player {
            1 => "Blue",
            2 => "Red",
            _ => "Unknown",
        })?;
        writeln!(f,"Available actions: {:?}",self.available_actions_ids().collect::<Vec<_>>())?;
        writeln!(f, "Game Over: {}", self.is_game_over)?;
        writeln!(
            f,
            "Direction codes: 0=Up, 1=Down, 2=Left, 3=Right, 4=Up-Left, 5=Up-Right, 6=Down-Left, 7=Down-Right"
        )?;

        Ok(())
    }
}
