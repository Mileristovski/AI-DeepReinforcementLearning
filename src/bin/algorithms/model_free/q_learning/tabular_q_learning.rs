use rand::prelude::IteratorRandom;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::environments::env::DeepDiscreteActionsEnv;
use std::fmt::Display;
use std::io;
use rand::SeedableRng;
use crate::config::DeepLearningParams;

pub fn episodic_tabular_q_learning<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default,
>(
    num_episodes: usize,
    episode_stop: usize,
    gamma: f32,
    alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
) -> [[f32; NUM_ACTIONS]; NUM_STATES] {
    // Initialize Q‑table to zero
    let mut q_table = [[0.0f32; NUM_ACTIONS]; NUM_STATES];
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let mut env = Env::default();
    env.set_against_random();
    
    let mut total_score = 0.0;

    for ep in 0..num_episodes {
        if ep > 0 && ep % episode_stop == 0 {
            println!("Mean Score : {:.3}", total_score / episode_stop as f32);
            total_score = 0.0;
        }
        let progress = ep as f32 / num_episodes as f32;
        let eps = (1.0 - progress) * start_epsilon + progress * final_epsilon;

        env.reset();
        let mut s = env.state_index();

        let mut a = if rand::random::<f32>() < eps {
            env.available_actions_ids().choose(&mut rng).unwrap()
        } else {
            // greedy over available
            env.available_actions_ids()
                .max_by(|&a1, &a2| q_table[s][a1].partial_cmp(&q_table[s][a2]).unwrap())
                .unwrap()
        };

        while !env.is_game_over() {
            let prev_score = env.score();
            env.step_from_idx(a);

            if env.is_game_over() {
                break;
            }
            
            let r = env.score() - prev_score;
            let s2 = env.state_index();

            let a2 = if rand::random::<f32>() < eps {
                env.available_actions_ids().choose(&mut rng).unwrap()
            } else {
                env.available_actions_ids()
                    .max_by(|&x, &y| q_table[s2][x].partial_cmp(&q_table[s2][y]).unwrap())
                    .unwrap()
            };

            // Q‑learning update
            let best_next = env
                .available_actions_ids()
                .map(|a| q_table[s2][a])
                .fold(f32::MIN, f32::max);
            let td_target = r + gamma * best_next;
            let td_error = td_target - q_table[s][a];
            q_table[s][a] += alpha * td_error;

            s = s2;
            a = a2;
        }
    }
    println!("Mean Score : {:.3}", total_score / episode_stop as f32);

    q_table
}

pub fn run_tabular_q_learning<
    const NUM_STATES_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATES_FEATURES, NUM_ACTIONS> + Display + Default,
>() {
    let params = DeepLearningParams::default();

    // Train
    let q_table = episodic_tabular_q_learning::<NUM_STATES_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(
        params.num_episodes,
        params.episode_stop,
        params.gamma,
        params.alpha,
        params.start_epsilon,
        params.final_epsilon,
    );

    // Test
    test_tabular::<NUM_STATES_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(&q_table);
}

pub fn test_tabular<
    const NUM_STATES_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATES_FEATURES, NUM_ACTIONS> + Display + Default,
>(
    q_table: &[[f32; NUM_ACTIONS]; NUM_STATES],
) {
    let mut env = Env::default();
    env.set_against_random();

    loop {
        env.reset();
        while !env.is_game_over() {
            println!("{}", env);
            let s = env.state_index();
            // pick the greedy action among available
            let action = env
                .available_actions_ids()
                .max_by(|&a1, &a2| q_table[s][a1].partial_cmp(&q_table[s][a2]).unwrap())
                .unwrap();
            env.step_from_idx(action);
        }
        println!("{}", env);
        println!("Press Enter to play another tabular‑Q episode or 'quit' to quit");
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).unwrap();
        if buf.trim().eq_ignore_ascii_case("quit")  {
            break
        }
    }
}
