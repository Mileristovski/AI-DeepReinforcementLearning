use std::fmt::Display;
use std::io;
use kdam::tqdm;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::config::DeepLearningParams;
use crate::environments::env::DeepDiscreteActionsEnv;

fn select_action_by_rollout<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: Clone + DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(
    env: &Env,
    rollouts_per_action: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> usize
where
    [(); NUM_STATE_FEATURES]:,
    [(); NUM_ACTIONS]:,
{
    let mut best_a = 0;
    let mut best_val = f32::NEG_INFINITY;

    for a in env.available_actions_ids() {
        let mut sum = 0.0;
        for _ in 0..rollouts_per_action {
            let mut sim = env.clone();
            sim.step_from_idx(a);
            while !sim.is_game_over() {
                let a2 = sim.available_actions_ids().choose(rng).unwrap();
                sim.step_from_idx(a2);
            }
            sum += sim.score();
        }
        let avg = sum / (rollouts_per_action as f32);
        if avg > best_val {
            best_val = avg;
            best_a = a;
        }
    }
    best_a
}

fn run_episode<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(
    env: &mut Env,
    rollouts_per_action: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> f32
where
    [(); NUM_STATE_FEATURES]:,
    [(); NUM_ACTIONS]:,
{
    while !env.is_game_over() {
        let best_a = select_action_by_rollout(env, rollouts_per_action, rng);
        env.step_from_idx(best_a);
    }
    env.score()
}

fn episodic_random_rollout<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: Clone + DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default,
>(
    num_episodes: usize,
    episode_stop: usize,
    rollouts_per_action: usize,
    rng: &mut Xoshiro256PlusPlus,
) where
    [(); NUM_STATE_FEATURES]:,
    [(); NUM_ACTIONS]:,
{
    let mut total_score = 0.0;
    let mut env = Env::default();
    env.set_against_random();

    for ep in tqdm!(0..num_episodes) {
        env.reset();
        total_score += run_episode(&mut env, rollouts_per_action, rng);

        if (ep + 1) % episode_stop == 0 {
            if ep > 0 && ep % episode_stop == 0 {
                println!("Mean Score : {:.3}", total_score / episode_stop as f32);
                total_score = 0.0;
            }
        }
    }
}

fn test_random_rollout<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Clone + Display + Default,
>(
    rollouts_per_action: usize,
    rng: &mut Xoshiro256PlusPlus,
) {
    loop {
        let mut env = Env::default();
        env.set_against_random();
        env.reset();

        println!("\n--- Test Episode ---\n");
        let score = run_episode(&mut env, rollouts_per_action, rng);

        println!("\nFinal state:\n{}\nScore: {}", env, score);
        println!("Press Enter to run another test, or type 'quit' to return.");
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).unwrap();
        if buf.trim().eq_ignore_ascii_case("quit") {
            break;
        }
    }
}

pub fn run_random_rollout<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Clone + Display + Default,
>() {
    let params = DeepLearningParams::default();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(params.rng_seed);

    episodic_random_rollout::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(
        params.num_episodes,
        params.episode_stop,
        params.rollouts_per_action,
        &mut rng,
    );
    test_random_rollout::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(
        params.rollouts_per_action,
        &mut rng,
    );
}
