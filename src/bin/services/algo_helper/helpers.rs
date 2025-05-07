use burn::prelude::*;
use rand::prelude::IteratorRandom;
use rand::Rng;

pub fn argmax(row: &Vec<f32>) -> usize {
    row.iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap()
        .0
}


pub fn max(row: &Vec<f32>) -> f32 {
    *row.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}


pub fn epsilon_greedy_action<B: Backend<FloatElem=f32, IntElem=i64>, const NUM_STATES_FEATURES: usize, const NUM_ACTIONS: usize>(
    q_s: &Tensor<B, 1>,
    mask_tensor: &Tensor<B, 1>,
    minus_one: &Tensor<B, 1>,
    plus_one:  &Tensor<B, 1>,
    fmin_vec:  &Tensor<B, 1>,
    available_actions: impl Iterator<Item=usize>,
    epsilon: f32,
    rng: &mut impl Rng
) -> usize {
    if rng.gen_range(0f32..=1f32) < epsilon {
        available_actions.choose(rng).unwrap()
    } else {
        let inverted_mask = mask_tensor.clone() * minus_one.clone() + plus_one.clone();
        let masked_q_s    = q_s.clone() * mask_tensor.clone()
            + inverted_mask * fmin_vec.clone();

        masked_q_s.argmax(0).into_scalar() as usize
    }
}

/*
pub fn epsilon_greedy_action(
    aa: DVector<i32>,          // Available actions
    q: &Vec<Vec<f32>>,   // Q-table
    state: usize,        // Current state
    epsilon: f32,        // Exploration probability
    rng: &mut impl rand::Rng, // Random number generator
) -> i32 {
    let rnd_number = rand::random::<f32>();

    if rnd_number <= epsilon {
        // Explore: Choose a random action
        *aa.as_slice().choose(rng).unwrap()
    } else {
        // Exploit: Choose the best action
        let mut best_a = 0;
        let mut best_a_score = f32::MIN;

        for &a in aa.iter() {
            let q_s_a = q[state][a as usize];

            if q_s_a >= best_a_score {
                best_a = a;
                best_a_score = q_s_a;
            }
        }
        best_a
    }
}
*/