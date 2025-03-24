use nalgebra::DVector;
use rand::Rng;
use crate::environments::env::Env;

pub struct GridEnv {
    s: DVector<i32>,
    a: Vec<i32>,
    r: DVector<f32>,
    p: Vec<Vec<Vec<Vec<f32>>>>,
    t: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    current_state: usize,
    current_score: f32
}

impl GridEnv {
    pub fn new() -> Self {
        let rows = 5;
        let cols = 5;
        let s = DVector::from_vec(vec![0; rows * cols]);
        let a = vec![0, 1, 2, 3];
        let r = DVector::from_vec(vec![-1.0, -0.1, 1.0]);
        let mut t = vec![0, (rows*cols)-1];
        for col in 1..cols {
            t.push(col*cols);
            t.push(col*cols-1);
        }
        let mut p = vec![vec![vec![vec![0.0; r.len()]; s.len()]; a.len()]; s.len()];

        for state in 0..s.len() {
            let (row, col) = (state / cols, state % cols);

            for action in 0..a.len() {
                // Determine the next state based on the action
                let (new_row, new_col) = match action {
                    0 if row > 0 => (row - 1, col),        // Up
                    1 if row < rows - 1 => (row + 1, col), // Down
                    2 if col > 0 => (row, col - 1),        // Left
                    3 if col < cols - 1 => (row, col + 1), // Right
                    _ => (row, col),
                };

                let new_state = new_row * cols + new_col;

                // Handle terminal states
                if state == 0 {
                    // Top-left: Terminal state with reward -1.0
                    p[state][action][state][0] = -1.0; // Reward index 0 corresponds to -1
                } else if state == s.len() - 1 {
                    // Bottom-right: Terminal state with reward +1.0
                    p[state][action][state][2] = 1.0; // Reward index 2 corresponds to +1
                } else {
                    // Non-terminal states
                    p[state][action][new_state][1] = -0.2; // Reward index 1 corresponds to 0
                }
            }
        }

        // For simplicity, start at middle cell if possible,
        // otherwise just start at 0.
        let starting_state = (rows/2) * cols + (cols/2);

        GridEnv {
            s,
            a,
            r,
            t,
            p,
            rows,
            cols,
            current_state: starting_state,
            current_score: 0.0
        }

    }

    /// Utility to convert a (row, col) pair into a single flattened index.
    pub fn rc_to_index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    /// Utility to convert a single flattened index into a (row, col) pair.
    pub fn index_to_rc(&self, idx: usize) -> (usize, usize) {
        let row = idx / self.cols;
        let col = idx % self.cols;
        (row, col)
    }
}

impl Env for GridEnv {
    fn num_states(&self) -> usize {
        self.s.len()
    }

    fn num_actions(&self) -> usize {
        self.a.len()
    }

    fn num_rewards(&self) -> usize {
        self.r.len()
    }

    fn get_reward_vector(&self) -> Vec<f32> {
        vec![-1.0f32, 0.0, 1.0]
    }

    fn get_terminal_states(&self) -> Vec<usize> {
        self.t.clone()
    }

    fn get_reward(&self, _num: usize) -> f32 {
        // You could define a more interesting reward structure.
        self.current_score
    }

    fn state_id(&self) -> Vec<i32> {
        vec![self.current_state as i32]
    }

    fn reset(&mut self) {
        // Reset to the center of the grid or 0 if there's no space.
        let grid_size = self.rows * self.cols;
        self.current_state = if grid_size > 0 { grid_size / 2 } else { 0 };
        self.current_score = 0.0;
    }

    fn display(&self) {
        let (agent_row, agent_col) = self.index_to_rc(self.current_state);
        for row in 0..self.rows {
            for col in 0..self.cols {
                if row == agent_row && col == agent_col {
                    print!("X ");
                } else {
                    print!("_ ");
                }
            }
            println!();
        }
        println!();
    }

    fn is_forbidden(&self, action: i32) -> bool {
        !self.available_actions().iter().any(|&x| x == action)
    }

    fn is_game_over(&self) -> bool {
        // The game is over if the current state is a terminal state.
        self.t.contains(&self.current_state)
    }

    fn available_actions(&self) -> Vec<i32> {
        if self.is_game_over() {
            vec![0]
        } else {
            self.a.clone()
        }
    }

    fn step(&mut self, action: i32) {
        // Check if the action is valid.
        if self.is_forbidden(action) {
            self.current_score -= 2.0;
            return;
        }

        if self.is_game_over() {
            panic!("Trying to play when the game is over!");
        }

        let (mut row, mut col) = self.index_to_rc(self.current_state);
        match action {
            0 => { // Up
                if row > 0 {
                    row -= 1;
                }
            }
            1 => { // Down
                if row < self.rows - 1 {
                    row += 1;
                }
            }
            2 => { // Left
                if col > 0 {
                    col -= 1;
                }
            }
            3 => { // Right
                if col < self.cols - 1 {
                    col += 1;
                }
            }
            _ => panic!("Invalid action"),
        };

        self.current_state = self.rc_to_index(row, col);
        if col == self.cols - 1 {
            self.current_score += self.r[2];
        } else if col == 0 {
            self.current_score += self.r[0];
        } else {
            self.current_score += self.r[1];
        }
    }

    fn score(&self) -> f32 { self.current_score }

    fn start_from_random_state(&mut self) {
        // Reset the game
        self.reset();

        // Make the player start from a random non-terminal location
        let mut rng = rand::thread_rng();
        let random_row = rng.gen_range(0..self.rows);
        let random_col = rng.gen_range(1..(self.cols-1));

        // Set the current state
        self.current_state = self.rc_to_index(random_row, random_col)
    }

    fn transition_probability(&self, s: usize, a: usize, s_p: usize, r_index: usize) -> f32 {
        self.p[s][a][s_p][r_index]
    }
}
