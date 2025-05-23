use kdam::tqdm;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::fmt::Display;
use std::io;
use std::time::Instant;
use crate::algorithms::model_based::monte_carlo::monte_carlo_tree_search_uct::mcts_search;
use crate::config::DeepLearningParams;
use crate::environments::env::DeepDiscreteActionsEnv;

fn mc_q<
    const N_S: usize,
    const N_A: usize,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Clone,
>(
    env: &Env,
    a: usize,
    rollouts: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> f32 {
    let mut sum = 0.0;
    for _ in 0..rollouts {
        let mut sim = env.clone();
        sim.step_from_idx(a);

        while !sim.is_game_over() {
            let ra = sim.available_actions_ids().choose(rng).unwrap();
            sim.step_from_idx(ra);
        }
        sum += sim.score();
    }
    sum / rollouts as f32
}

fn select_action_by_rollout<
    const N_S: usize,
    const N_A: usize,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Clone,
>(
    env: &Env,
    rollouts_per_action: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> usize {
    let mut best_a   = usize::MAX;         // sentinel
    let mut best_val = f32::NEG_INFINITY;

    for a in env.available_actions_ids() {
        let q = mc_q(env, a, rollouts_per_action, rng);
        if q > best_val {
            best_val = q;
            best_a   = a;
        }
    }
    debug_assert!(best_a != usize::MAX, "no legal action?");
    best_a
}

fn run_episode<
    const N_S: usize,
    const N_A: usize,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Clone,
>(
    env: &mut Env,
    rollouts_per_action: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> f32 {
    while !env.is_game_over() {
        let a = select_action_by_rollout(env, rollouts_per_action, rng);
        env.step_from_idx(a);
    }
    env.score()
}

fn episodic_random_rollout<
    const N_S: usize,
    const N_A: usize,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Clone + Default + Display,
>(
    episodes: usize,
    log_every: usize,
    rollouts_per_action: usize,
    rng: &mut Xoshiro256PlusPlus,
) {
    let mut env = Env::default();

    let mut score_sum = 0.0;
    for ep in tqdm!(0..=episodes) {
        if ep > 0 && ep % log_every == 0 {
            println!("MCRR - Mean Score : {:.3}", score_sum / log_every as f32);
            score_sum = 0.0;
        }
        env.reset();
        score_sum += run_episode(&mut env, rollouts_per_action, rng);
    }
}

fn test_random_rollout<
    const N_S: usize,
    const N_A: usize,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Clone + Default + Display,
>(
    params: &DeepLearningParams,
    rng: &mut Xoshiro256PlusPlus,
) {
        let mut env = Env::default();
        let mut total = 0.0;
        let game_start = Instant::now();

        for _ in 0..params.num_episodes {
            env.reset();
            while !env.is_game_over() {
                let action = mcts_search(&env, params.mcts_simulations, params.mcts_c, rng);
                env.step_from_idx(action);
            }
            total += env.score();

        }

        let mean_total = total / params.num_episodes as f32;
        let mean_duration = game_start.elapsed().as_secs_f64() / params.num_episodes as f64;

        println!("MCRR - Mean Score : {:.3}, Mean Duration : {:.3}s", mean_total * 100.0, mean_duration);
        let mut buf = String::new();
        println!("Press Enter to continue...");
        io::stdin().read_line(&mut buf).unwrap();
}

pub fn run_random_rollout<
    const N_S: usize,
    const N_A: usize,
    Env: DeepDiscreteActionsEnv<N_S, N_A> + Clone + Default + Display,
>(_env_name: &str,p: &DeepLearningParams) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(p.rng_seed);

    episodic_random_rollout::<N_S, N_A, Env>(
        p.num_episodes,
        p.episode_stop,
        p.mcts_rollouts_per_action,
        &mut rng,
    );
    if p.run_test {
        test_random_rollout::<N_S, N_A, Env>(p, &mut rng);
    }
}
