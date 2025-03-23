use nalgebra::DVector;
use crate::environments::env::Env;
use std::collections::HashMap;

pub struct TicTacToeEnv {
    board: [char; 9],
    actions: DVector<i32>,
    current_player: char,
    pub current_state: usize,
    pub game_over: bool,
    pub winner: Option<char>,
    terminal_states: [[i32; 3]; 8],
    current_score: f32,
    transition_matrix: Option<Vec<Vec<Vec<Vec<f32>>>>>,
}

impl TicTacToeEnv {
    pub fn new() -> Self {
        let terminal_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
            [0, 4, 8], [2, 4, 6],           // Diagonals
        ];
        let board = [' '; 9];
        let actions = DVector::from_vec((0..9).collect());
        let current_player = 'X';
        let current_state = 0;
        let game_over = false;
        let winner = None;
        let current_score = 0.0;

        // Create a temporary instance first
        Self {
            board,
            actions,
            current_player,
            current_state,
            game_over,
            winner,
            terminal_states,
            current_score,
            transition_matrix: None,
        }
    }

    fn generate_transition_matrix(&self) -> Vec<Vec<Vec<Vec<f32>>>> {
        const STATE_COUNT: usize = 3usize.pow(9);
        const ACTION_COUNT: usize = 9;
        const REWARD_COUNT: usize = 2;

        // Initialize a 4D array filled with zeros
        let mut transitions = vec![vec![vec![vec![0.0; REWARD_COUNT]; STATE_COUNT]; ACTION_COUNT]; STATE_COUNT];
        let mut state_map = HashMap::new();
        let mut index = 0;

        // Generate all valid board states
        for i in 0..STATE_COUNT {
            let board = Self::state_to_board(i);
            if Self::is_valid_board(&board) {
                state_map.insert(board.clone(), index);
                index += 1;
            }
        }

        // Populate the transition probabilities
        for (&board, &s) in &state_map {
            for a in 0..ACTION_COUNT {
                if board[a] == ' ' {
                    let mut new_board = board.clone();
                    new_board[a] = 'X';
                    if let Some(&s_p) = state_map.get(&new_board) {
                        transitions[s][a][s_p][0] = 1.0;

                        // Check if the move results in a win
                        if Self::check_winner(&self, &new_board) == Some('X') {
                            transitions[s][a][s_p][1] = 1.0;
                        }
                    }
                }
            }
        }

        transitions
    }

    fn state_to_board(mut state: usize) -> [char; 9] {
        let mut board = [' '; 9];
        for i in (0..9).rev() {
            match state % 3 {
                0 => board[i] = ' ',
                1 => board[i] = 'X',
                2 => board[i] = 'O',
                _ => unreachable!(),
            }
            state /= 3;
        }
        board
    }

    fn is_valid_board(board: &[char; 9]) -> bool {
        let x_count = board.iter().filter(|&&c| c == 'X').count();
        let o_count = board.iter().filter(|&&c| c == 'O').count();
        x_count == o_count || x_count == o_count + 1
    }

    pub fn print_transition_matrix(&self, num_states: usize) {
        for s in 0..num_states {
            for a in 0..9 {
                for s_p in 0..num_states {
                    for r in 0..2 {
                        let value = self.transition_matrix.as_ref().unwrap()[s][a][s_p][r];
                        if value > 0.0 {
                            println!("T({}, {}, {}, {}) = {:.1}", s, a, s_p, r, value);
                        }
                    }
                }
            }
        }
    }

    fn check_winner(&self, board: &[char; 9]) -> Option<char> {
        for state in &self.terminal_states {
            let indices: Vec<usize> = state.iter().map(|&x| x as usize).collect(); // Convert to usize
            if board[indices[0]] != ' ' && board[indices[0]] == board[indices[1]] && board[indices[1]] == board[indices[2]] {
                return Some(board[indices[0]]);
            }
        }
        None
    }
}

impl Env for TicTacToeEnv {
    fn num_states(&self) -> usize { 9 }

    fn num_actions(&self) -> usize { self.actions.len() }

    fn num_rewards(&self) -> usize { 2 }

    fn get_reward_vector(&self) -> Vec<f32> { vec![-1.0, 1.0] }

    fn get_terminal_states(&self) -> Vec<usize> { vec![] }

    fn get_reward(&self, _num: usize) -> f32 { if self.winner == Some('X') { 1.0 } else { -1.0 } }

    fn state_id(&self) -> usize { self.current_state }

    fn reset(&mut self) {
        self.board = [' '; 9];
        self.current_player = 'X';
        self.game_over = false;
        self.winner = None;
    }

    fn display(&self) {
        for row in 0..3 {
            println!("{} | {} | {}", self.board[row * 3], self.board[row * 3 + 1], self.board[row * 3 + 2]);
            if row < 2 { println!("- + - + -"); }
        }
        println!();
    }

    fn is_forbidden(&self, action: i32) -> bool {
        self.board[action as usize] != ' '
    }

    fn is_game_over(&self) -> bool { self.game_over }

    fn available_actions(&self) -> DVector<i32> { DVector::from_vec(self.board.iter().enumerate().filter(|&(_, &c)| c == ' ').map(|(i, _)| i as i32).collect()) }

    fn step(&mut self, action: i32) {
        if self.game_over || self.is_forbidden(action) {
            return;
        }

        if self.current_player == 'X' {
            self.current_score -= 0.1;
        }

        self.board[action as usize] = self.current_player;

        for &state in &self.terminal_states {
            if self.board[state[0] as usize] != ' '
                && self.board[state[0] as usize] == self.board[state[1] as usize]
                && self.board[state[1] as usize] == self.board[state[2] as usize] {
                self.winner = Some(self.board[state[0] as usize]);
                self.game_over = true;
            }
        }

        if !self.board.contains(&' ') {
            self.game_over = true;
        }

        if self.game_over && self.winner == Option::from('X') {
            self.current_score += 1.0;
        } else if self.game_over && self.winner == Option::from('O') {
            self.current_score -= 1.0;
        } else if self.game_over {
            self.current_score -= 0.5;
        }

        self.current_player = if self.current_player == 'X' { 'O' } else { 'X' };
    }

    fn score(&self) -> f32 { self.current_score }

    fn start_from_random_state(&mut self) { self.reset() }

    fn transition_probability(&self, _s: usize, _a: usize, _s_p: usize, _r_index: usize) -> f32 {
        unimplemented!("Slows down application too much")
    }
}
