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
    terminal_states: Vec<usize>,
    score: f32,
    pub blue_player: usize,
    pub red_player: usize,
    bobail_move: bool,
    directions:  [(isize, isize, isize); 8],
    empty: f32,
    chip_lookup: HashMap<usize, usize>,
    pub is_random_state: bool,
    against_random: bool,
    actions_lookup: [usize; BB_NUM_ACTIONS],
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
        
        let actions_lookup: [usize; BB_NUM_ACTIONS] = {
            let mut actions = [0; BB_NUM_ACTIONS];
            let chip_ids = [11, 1, 2, 3, 4, 5];

            let mut i = 0;
            for &chip_id in chip_ids.iter() {
                for dir in 0..8 {
                    actions[i] = chip_id * 10 + dir;
                    i += 1;
                }
            }
            actions
        };
        
        Self {
            current_player: 1,
            previous_player: 1,
            board,
            is_game_over: false,
            terminal_states: vec!(0usize, 1, 2, 3, 4, 20, 21, 22, 23, 24),
            score: 0.0,
            blue_player: 1,
            red_player: 2,
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
            actions_lookup
        }
    }

    fn decode_action(&self, action: usize) -> (usize, usize, usize, usize) {
        let chip_id   = action / 10;       // 1-11
        let direction = action % 10;       // 0-7

        let index = *self.chip_lookup
            .get(&chip_id)
            .expect("chip_id missing from lookup");

        let row = index / COLS;
        let col = index % COLS;

        (chip_id, row, col, direction)
    }

    fn get_possible_moves(&self, row: usize, col: usize) -> Vec<usize> {
        let mut possible_moves = Vec::new();

        for &(dr, dc, direction) in &self.directions {
            let new_row = row as isize + dr;
            let new_col = col as isize + dc;

            // Check if the new position is within bounds
            if new_row >= 0 && new_row < ROWS as isize && new_col >= 0 && new_col < COLS as isize {
                let index = (new_row as usize) * COLS + (new_col as usize);
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
                    || next_row >= ROWS as isize
                    || next_col < 0
                    || next_col >= COLS as isize
                {
                    break;
                }

                let index = (next_row as usize) * COLS + next_col as usize;
                if self.board[index] != self.empty {
                    break;
                }

                new_row = next_row;
                new_col = next_col;
            }
        }

        let dst_index = new_row as usize * COLS + new_col as usize;
        let src_index = row * COLS + col;

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
        let mut action_ids = Vec::new();

        // Chip ID order must match the mask layout
        let controlled_chip_ids: Vec<usize> = if self.bobail_move {
            vec![11]
        } else {
            match self.current_player {
                1 => vec![1, 2, 3, 4, 5],       // Blue player
                _ => vec![]
            }
        };

        for (chip_index, &chip_id) in controlled_chip_ids.iter().enumerate() {
            if let Some(&index) = self.chip_lookup.get(&chip_id) {
                // Check if this chip is allowed to move right now
                let is_bobail_turn = self.bobail_move && chip_id == 11;
                let is_player_turn = !self.bobail_move && matches!(self.current_player, 1) && (1..=5).contains(&chip_id);

                if is_bobail_turn || is_player_turn {
                    let row = index / COLS;
                    let col = index % COLS;

                    for dir in self.get_possible_moves(row, col) {
                        let encoded_action_id = chip_index * 8 + dir;
                        action_ids.push(encoded_action_id);
                    }
                }
            }
        }

        action_ids.into_iter()
    }
    
    fn available_actions(&self) -> impl Iterator<Item=usize> {         
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
                let row = index / COLS;
                let col = index % COLS;

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
                    let row = index / COLS;
                    let col = index % COLS;

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
        Self::move_piece(self, action);

        self.bobail_move = !self.bobail_move;

        // Verify if the game is over
        let bobail_index = self.chip_lookup.get(&11).copied();
        let bobail_in_terminal = match bobail_index {
            Some(idx) => self.terminal_states.contains(&idx),
            None => false,
        };

        let no_actions_left = self.available_actions().next().is_none();

        // Check if game is over
        if bobail_in_terminal || no_actions_left {
            self.is_game_over = true;

            // Assign winner and score
            if let Some(idx) = bobail_index {
                if idx < 5 {
                    self.score -= 1.0;
                } else if idx > 19 {
                    self.score += 1.0;
                }
            }

            if no_actions_left {
                if self.current_player == self.blue_player {
                    self.score += 1.0;
                } else {
                    self.score -= 1.0;
                }
            }
        }

        if self.bobail_move && !self.is_game_over {
            self.current_player = if self.current_player == self.blue_player {
                self.red_player
            } else {
                self.blue_player
            };
        }

        if self.against_random && self.current_player == self.red_player && !self.is_game_over {
            // random move
            let mut rng = thread_rng();
            let random_action = self.available_actions().choose(&mut rng).unwrap();
            self.step(random_action)
        }
    }

    fn is_game_over(&self) -> bool { self.is_game_over }

    fn score(&self) -> f32 { self.score }

    fn reset(&mut self) {
        let mut board = [0.0f32; NUM_BOARD_SIZE];
        for red in 0..COLS {
            board[red] = red as f32 + 6.0;
        }
        // Add the red player
        for blue in NUM_BOARD_SIZE-COLS..NUM_BOARD_SIZE {
            board[blue] = blue as f32 - ((NUM_BOARD_SIZE-COLS) as f32 - 1.0);
        }

        self.chip_lookup = HashMap::from([
            // blue
            (6,  0), (7,  1), (8,  2), (9,  3), (10,  4),
            // red
            (1, 20), (2, 21), (3, 22), (4, 23), (5, 24),
            // bobail
            (11, NUM_BOARD_SIZE / 2),
        ]);
        // Add the bobail
        board[NUM_BOARD_SIZE/2] = 11.0;
        
        self.board = board;
        
        self.current_player = 1;
        self.previous_player = 1;
        self.is_game_over = false;
        self.score = 0.0;
        self.bobail_move = false;
    }

    fn set_from_random_state(&mut self) { self.is_random_state = !self.is_random_state }

    fn set_against_random(&mut self) -> bool {
        self.against_random = !self.against_random;
        self.against_random
    }

    fn step_from_idx(&mut self, action_idx: usize) {
        if action_idx >= BB_NUM_ACTIONS {
            eprintln!("Invalid action index: {}", action_idx);
            return;
        }

        let action = self.actions_lookup[action_idx];
        self.step(action);
    }

    fn state_index(&self) -> usize {
        panic!("Can't be implemented!")
    }
}

impl Display for BobailHeuristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, " {}", "----+".repeat(COLS - 1) + "----")?;

        for row in 0..ROWS {
            write!(f, "|")?;
            for col in 0..COLS {
                let idx = row * COLS + col;
                let val = self.board[idx] as usize;
                let cell = Self::display_helper(val);
                write!(f, "{}|", cell)?;
            }
            writeln!(f)?;
            writeln!(f, " {}", "----+".repeat(COLS - 1) + "----")?;
        }

        writeln!(f)?;
        writeln!(f, "Score: {}", self.score)?;
        writeln!(f, "Player to move: {}", match self.current_player {
            1 => "Blue",
            2 => "Red",
            _ => "Unknown",
        })?;
        writeln!(f,"Available actions: {:?}",self.available_actions().collect::<Vec<_>>())?;
        writeln!(f, "Game Over: {}", self.is_game_over)?;
        writeln!(
            f,
            "Direction codes: 0=Up, 1=Down, 2=Left, 3=Right, 4=Up-Left, 5=Up-Right, 6=Down-Left, 7=Down-Right"
        )?;

        Ok(())
    }
}
