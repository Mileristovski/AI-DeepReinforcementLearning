use colored::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use regex::Regex;

use crate::environments::env::Env;

pub struct BobailEnv {
    pub board: [usize; 25],
    pub current_player: usize,
    pub previous_player: usize,
    game_over: bool,
    winner: usize,
    terminal_states: Vec<usize>,
    current_score: f32,
    r: Vec<f32>,
    rows: usize,
    cols: usize,
    pub blue_player: usize,
    pub red_player: usize,
    pub bobail: usize,
    bobail_move: bool,
    directions:  [(isize, isize, isize); 8],
    empty: usize
}

impl BobailEnv {
    pub fn new() -> Self {
        let terminal_states = vec!(0usize, 1, 2, 3, 4, 20, 21, 22, 23, 24);
        const ROWS: usize = 5;
        const COLS: usize = 5;
        let blue_player = 1;
        let red_player = 2;
        let bobail = 3;
        let empty = 0;
        let mut board: [usize; ROWS * COLS] = [0; ROWS * COLS];
        let current_player = blue_player.clone();
        let previous_player = blue_player.clone();
        let game_over = false;
        let winner = 0;
        let current_score = 0.0;
        let r = vec![-1.0f32, 0.0, 1.0];
        let bobail_move = false;
        let directions = [
            (-1, 0, 0),     // Up
            (1, 0, 1),      // Down
            (0, -1, 2),     // Left
            (0, 1, 3),      // Right
            (-1, -1, 4),    // Up-Left
            (-1, 1, 5),     // Up-Right
            (1, -1, 6),     // Down-Left
            (1, 1, 7),      // Down-Right
        ];


        let mut env = Self {
            current_player,
            previous_player,
            board,
            game_over,
            winner,
            terminal_states,
            current_score,
            r,
            rows: ROWS,
            cols: COLS,
            blue_player,
            red_player,
            bobail,
            bobail_move,
            directions,
            empty
        };

        env.init_board();

        env
    }

    fn init_board(&mut self) {
        // Add the blue player
        for red in 0..self.cols {
            self.board[red] = self.red_player.clone();
        }

        // Add the red player
        for blue in self.board.len()- self.cols..self.board.len() {
            self.board[blue] = self.blue_player.clone();
        }

        // Add the bobail
        self.board[self.board.len()/2] = self.bobail.clone();
    }

    pub fn get_row_col(action: i32) -> (usize, usize, usize) {
        let current_row = action/100;
        let current_col = action%100/10;
        let direction = action%10;

        ((current_row - 1) as usize, (current_col - 1) as usize, direction as usize)
    }

    fn get_possible_moves(&self, row: usize, col: usize) -> Vec<isize> {
        let mut possible_moves = Vec::new();

        for &(dr, dc, direction) in &self.directions {
            let new_row = row as isize + dr;
            let new_col = col as isize + dc;

            // Check if the new position is within bounds
            if new_row >= 0 && new_row < self.rows as isize && new_col >= 0 && new_col < self.cols as isize {
                let index = (new_row as usize) * self.cols + (new_col as usize);
                if self.board[index] == self.empty { // Only move if the spot is empty
                    possible_moves.push(direction);
                }
            }
        }
        possible_moves
    }

    fn move_piece(&mut self, action: i32) -> (usize, usize) {
        let ( mut row, mut col, dir) = Self::get_row_col(action);
        let mut new_row = 0;
        let mut new_col = 0;

        // Move as far as possible in the given direction
        let correct_direction = self.directions.iter().find(|&&(dr, dc, direction)| direction == dir as isize);

        if let Some(&(dr, dc, name)) = &correct_direction {
            if self.bobail_move {
                new_row = row as isize + dr;
                new_col = col as isize + dc;
            } else {
                new_row = row as isize;
                new_col = col as isize;
                // Move as far as possible in the given direction
                loop {
                    let next_row = new_row + dr;
                    let next_col = new_col + dc;

                    // Check if the new position is out of bounds
                    if next_row < 0 || next_row >= self.rows as isize || next_col < 0 || next_col >= self.cols as isize {
                        break; // Stop at the board edge
                    }

                    let index = (next_row as usize) * self.cols + (next_col as usize);

                    // Stop if there's an obstacle
                    if self.board[index] != self.empty {
                        break;
                    }

                    // Update position
                    new_row = next_row;
                    new_col = next_col;
                }
            }

            // Check if the new position is within bounds
            let index = (new_row as usize) * self.cols + (new_col as usize);
            if self.board[index] == self.empty {
                self.board[row * self.rows + col] = self.empty;
                self.board[new_row as usize * self.rows + new_col as usize] = self.current_player.clone();
            }
        } else {
            panic!("Something went wrong");
        }

        (new_row as usize, new_col as usize)
    }

    fn clean_string(s: &str) -> String {
        let ansi_escape = Regex::new("\x1b\\[[0-9;]*m").unwrap();
        ansi_escape.replace_all(s, "").to_string()
    }

    pub fn get_input_vec(&self) -> Vec<String> {
        let mut state_bin = vec![];
        for state in 0..self.board.len()*4 {
            let binary_str = format!("{:b}", state);
            state_bin.push(binary_str);
        }

        state_bin
    }
    
    pub fn get_output_vec(&mut self) -> Vec<i32> {
        let old_board = self.board.clone();
        self.board = [0; 25];

        let mut state_vec = vec![];

        for i in 0..self.board.len() {
            self.board[i] = self.blue_player;
            for action in self.available_actions() {
                state_vec.push(action);
            }
            self.board[i] = self.empty;
        }

        self.board = old_board;
        state_vec
    }

    fn display_helper(num: usize) -> ColoredString {
        let res = match num {
            1 => Colorize::blue("B"),
            2 => Colorize::red("R"),
            3 => Colorize::yellow("Y"),
            _ => Colorize::normal(" "),
        };

        res
    }
}

impl Env for BobailEnv {
    fn num_states(&self) -> usize { self.board.len() }

    fn num_actions(&self) -> usize { self.get_input_vec().len() }

    fn num_rewards(&self) -> usize { self.r.len() }

    fn get_reward_vector(&self) -> Vec<f32> { self.r.clone() }

    fn get_terminal_states(&self) -> Vec<usize> { self.terminal_states.clone() }

    fn get_reward(&self, _num: usize) -> f32 { self.r[_num] }

    fn state_id(&self) -> Vec<i32> {
        self.board.iter().map(|&x| x as i32).collect()
    }

    fn reset(&mut self) {
        self.board = core::array::from_fn(|_| self.empty);
        self.init_board();
        self.current_player = self.blue_player.clone();
        self.game_over = false;
        self.winner = self.empty;
        self.bobail_move = false;
    }

    fn display(&self) {
        println!("- + - + - + - + -");
        for row in 0..5 {
            println!("{} | {} | {} | {} | {}",
                     Self::display_helper(self.board[row * self.cols]),
                     Self::display_helper(self.board[row * self.cols + 1]),
                     Self::display_helper(self.board[row * self.cols + 2]),
                     Self::display_helper(self.board[row * self.cols + 3]),
                     Self::display_helper(self.board[row * self.cols + 4]));
            if row < self.cols { println!("- + - + - + - + -"); }
        }
        println!();
    }

    fn is_forbidden(&self, action: i32) -> bool {
        !self.available_actions().iter().any(|&x| x == action)
    }

    fn is_game_over(&self) -> bool {
        let bobail_index: Vec<usize> = self.board.iter()
            .enumerate()
            .filter_map(|(idx, c)| if *c == self.bobail { Some(idx) } else { None })
            .collect();

        self.terminal_states.iter().any(|&x| x == bobail_index[0]) || self.available_actions().is_empty() ||self.game_over
    }

    fn available_actions(&self) -> Vec<i32> {
        let mut available_action = Vec::new();

        let indexes: Vec<usize> = self.board.iter()
            .enumerate()
            .filter_map(|(idx, c)| if *c == self.current_player { Some(idx) } else { None })
            .collect();

        for &elem in indexes.iter() {
            let mut row = (elem / self.rows) as i32;
            let mut col = (elem % self.cols) as i32;
            let mut all_moves = Self::get_possible_moves(&self, row as usize, col as usize);
            for action in all_moves.iter() {
                available_action.push((row+1)*100 + (col+1)*10 + *action as i32);
            }
        }

        available_action
    }

    fn step(&mut self, action: i32) {
        if self.game_over || self.is_forbidden(action) {
            return;
        }

        // Move the piece
        let (new_row, _) = Self::move_piece(self, action);

        // Change player
        self.current_player = if self.current_player == self.bobail && self.previous_player == self.blue_player {
            self.bobail_move = false;
            self.red_player.clone()
        } else if self.current_player == self.bobail && self.previous_player == self.red_player {
            self.bobail_move = false;
            self.blue_player.clone()
        } else {
            self.bobail_move = true;
            self.previous_player = self.current_player.clone();
            self.bobail.clone()
        };

        // Verify if the game is over
        if self.is_game_over() {
            self.winner = if new_row == 0 {
                self.red_player.clone()
            } else if new_row == self.rows-1 {
                self.blue_player.clone()
            } else {
                self.previous_player.clone()
            };
            self.game_over = true;
        }

        // Add the rewards to the score
        if self.game_over && self.winner == self.blue_player {
            self.current_score += self.get_reward(2);
        } else if self.game_over && self.winner == self.red_player {
            self.current_score += self.get_reward(0);
        } else if self.current_player == self.blue_player && !self.bobail_move {
            self.current_score += self.get_reward(1); // Add a movement penalty
        }
    }

    fn score(&self) -> f32 { self.current_score }

    fn start_from_random_state(&mut self) {
        self.board = core::array::from_fn(|_| self.empty);
        let mut rng = thread_rng();

        // Get all board positions except first and last row for bobail
        let valid_positions: Vec<usize> = (self.cols..self.board.len() - self.cols).collect();

        // Place bobail randomly
        if let Some(&bobail_pos) = valid_positions.choose(&mut rng) {
            self.board[bobail_pos] = self.bobail.clone();
        }

        // Get all empty positions
        let mut empty_positions: Vec<usize> = self.board.iter()
            .enumerate()
            .filter_map(|(idx, c)| if *c == self.empty { Some(idx) } else { None })
            .collect();

        empty_positions.shuffle(&mut rng);

        // Place 5 blue pieces
        for _ in 0..5 {
            if let Some(pos) = empty_positions.pop() {
                self.board[pos] = self.blue_player.clone();
            }
        }

        // Place 5 red pieces
        for _ in 0..5 {
            if let Some(pos) = empty_positions.pop() {
                self.board[pos] = self.red_player.clone();
            }
        }

        self.current_player = self.bobail.clone();
        self.previous_player = self.red_player.clone();
        self.game_over = false;
        self.winner = self.empty;
        self.bobail_move = false;
    }

    fn transition_probability(&self, _s: usize, _a: usize, _s_p: usize, _r_index: usize) -> f32 {
        unimplemented!("Slows down application too much")
    }
}
