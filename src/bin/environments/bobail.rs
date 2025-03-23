use nalgebra::DVector;
use crate::environments::env::Env;
use colored::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct BobailEnv {
    pub board: [ColoredString; 25],
    actions: DVector<i32>,
    current_player: ColoredString,
    previous_player: ColoredString,
    current_state: usize,
    game_over: bool,
    winner: ColoredString,
    terminal_states: Vec<usize>,
    current_score: f32,
    r: Vec<f32>,
    rows: usize,
    cols: usize,
    blue_player: ColoredString,
    red_player: ColoredString,
    bobeil: ColoredString,
    bobeil_move: bool,
}

impl BobailEnv {
    pub fn new() -> Self {
        let terminal_states = vec!(0usize, 1, 2, 3, 4, 20, 21, 22, 23, 24);
        const ROWS: usize = 5;
        const COLS: usize = 5;
        let blue_player = Colorize::blue("B");
        let red_player = Colorize::red("R");
        let bobeil = Colorize::yellow("Y");
        let mut board: [ColoredString; ROWS * COLS] = std::array::from_fn(|_| " ".normal());
        let actions = DVector::from_vec((0..8).collect());
        let current_player = blue_player.clone();
        let previous_player = blue_player.clone();
        let current_state = 0;
        let game_over = false;
        let winner = "0".normal();
        let current_score = 0.0;
        let r = vec![-1.0f32, -0.1, 1.0];
        let bobeil_move = false;


        let mut env = Self {
            board,
            actions,
            current_player,
            previous_player,
            current_state,
            game_over,
            winner,
            terminal_states,
            current_score,
            r,
            rows: ROWS,
            cols: COLS,
            blue_player,
            red_player,
            bobeil,
            bobeil_move
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
        self.board[self.board.len()/2] = self.bobeil.clone();
    }

    pub fn get_row_col(action: i32) -> (usize, usize, usize, usize) {
        let current_row = action/1000;
        let current_col = action%1000/100;
        let new_row = action%100/10;
        let new_col = action%10;

        ((current_row - 1) as usize, (current_col - 1) as usize, (new_row - 1) as usize, (new_col - 1) as usize)
    }

    fn get_possible_moves(&self, row: usize, col: usize, bobail: bool) -> Vec<Vec<i32>> {
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

        let mut possible_moves = Vec::new();
        let mut new_row;
        let mut new_col;

        for &(dr, dc, name) in &directions {
            if bobail {
                new_row = row as isize + dr;
                new_col = col as isize + dc;
            } else {
                // Move as far as possible in the given direction
                new_row = row as isize;
                new_col = col as isize;

                loop {
                    let next_row = new_row + dr;
                    let next_col = new_col + dc;

                    // Check if the new position is out of bounds
                    if next_row < 0 || next_row >= self.rows as isize || next_col < 0 || next_col >= self.cols as isize {
                        break; // Stop at the board edge
                    }

                    let index = (next_row as usize) * self.cols + (next_col as usize);

                    // Stop if there's an obstacle
                    if self.board[index] != " ".normal() {
                        break;
                    }

                    // Update position
                    new_row = next_row;
                    new_col = next_col;
                }
            }

            // Check if the new position is within bounds
            if new_row >= 0 && new_row < self.rows as isize && new_col >= 0 && new_col < self.cols as isize {
                let index = (new_row as usize) * self.cols + (new_col as usize);
                if self.board[index] == " ".normal() { // Only move if the spot is empty
                    possible_moves.push(vec![new_row as i32, new_col as i32]);
                }
            }
        }

        possible_moves
    }
}

impl Env for BobailEnv {
    fn num_states(&self) -> usize { self.board.len() }

    fn num_actions(&self) -> usize { self.actions.len() }

    fn num_rewards(&self) -> usize { self.r.len() }

    fn get_reward_vector(&self) -> Vec<f32> { self.r.clone() }

    fn get_terminal_states(&self) -> Vec<usize> { self.terminal_states.clone() }

    fn get_reward(&self, _num: usize) -> f32 { self.r[_num] }

    fn state_id(&self) -> usize { self.current_state }

    fn reset(&mut self) {
        self.board = core::array::from_fn(|_| " ".normal());
        self.init_board();
        self.current_player = self.blue_player.clone();
        self.game_over = false;
        self.winner = "0".normal();
        self.bobeil_move = false;
    }

    fn display(&self) {
        println!("- + - + - + - + -");
        for row in 0..5 {
            println!("{} | {} | {} | {} | {}", self.board[row * self.cols], self.board[row * self.cols + 1], self.board[row * self.cols + 2], self.board[row * self.cols + 3], self.board[row * self.cols + 4]);
            if row < self.cols { println!("- + - + - + - + -"); }
        }
        println!();
    }

    fn is_forbidden(&self, action: i32) -> bool {
        !self.available_actions().iter().any(|&x| x == action)
    }

    fn is_game_over(&self) -> bool {
        let bobeil_index: Vec<usize> = self.board.iter()
            .enumerate()
            .filter_map(|(idx, c)| if *c == self.bobeil { Some(idx) } else { None })
            .collect();

        self.terminal_states.iter().any(|&x| x == bobeil_index[0]) || self.available_actions().is_empty()
    }

    fn available_actions(&self) -> DVector<i32> {
        let mut available_action = Vec::new();

        let indexes: Vec<usize> = self.board.iter()
            .enumerate()
            .filter_map(|(idx, c)| if *c == self.current_player { Some(idx) } else { None })
            .collect();

        for &elem in indexes.iter() {
            let mut row = (elem / self.rows) as i32;
            let mut col = (elem % self.cols) as i32;
            let mut all_moves = Self::get_possible_moves(&self, row as usize, col as usize, self.bobeil_move);
            for action in all_moves.iter() {
                available_action.push((row+1)*1000 + (col+1)*100 + (action[0]+1)*10 + action[1]+1);
            }
        }

        DVector::from_vec(available_action)
    }

    fn step(&mut self, action: i32) {
        if self.is_forbidden(action) {
            return;
        }

        let (row, col, new_row, new_col) = Self::get_row_col(action);

        // Move the piece
        self.board[row*self.cols + col] = " ".normal();
        self.board[new_row*self.cols + new_col] = self.current_player.clone();

        // Change player
        self.current_player = if self.current_player == self.bobeil && self.previous_player == self.blue_player {
            self.bobeil_move = false;
            self.red_player.clone()
        } else if self.current_player == self.bobeil && self.previous_player == self.red_player {
            self.bobeil_move = false;
            self.blue_player.clone()
        } else {
            self.bobeil_move = true;
            self.previous_player = self.current_player.clone();
            self.bobeil.clone()
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
        } else if self.previous_player == self.blue_player {
            self.current_score += self.get_reward(1); // Add a movement penalty
        }
    }

    fn score(&self) -> f32 { self.current_score }

    fn start_from_random_state(&mut self) {
        self.board = core::array::from_fn(|_| " ".normal());
        let mut rng = thread_rng();

        // Get all board positions except first and last row for bobeil
        let valid_positions: Vec<usize> = (self.cols..self.board.len() - self.cols).collect();

        // Place bobeil randomly
        if let Some(&bobeil_pos) = valid_positions.choose(&mut rng) {
            self.board[bobeil_pos] = self.bobeil.clone();
        }

        // Get all empty positions
        let mut empty_positions: Vec<usize> = self.board.iter()
            .enumerate()
            .filter_map(|(idx, c)| if *c == " ".normal() { Some(idx) } else { None })
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

        self.current_player = self.blue_player.clone();
        self.game_over = false;
        self.winner = "0".normal();
        self.bobeil_move = false;
    }

    fn transition_probability(&self, _s: usize, _a: usize, _s_p: usize, _r_index: usize) -> f32 {
        unimplemented!("Slows down application too much")
    }
}
