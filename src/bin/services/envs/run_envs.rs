use crate::environments::env::DeepDiscreteActionsEnv;
use crate::gui::cli::common::{end_of_run, reset_screen};
use std::fmt::Display;
use std::io;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;
use rand::prelude::IteratorRandom;
// model-free
use crate::algorithms::model_free::sarsa::run_episodic_semi_gradient_sarsa;
// use crate::algorithms::model_free::q_learning::tabular_q_learning::run_tabular_q_learning;
use crate::algorithms::model_free::q_learning::deep_q_learning::run_deep_q_learning;
use crate::algorithms::model_free::q_learning::double_deep_q_learning::run_double_deep_q_learning;
use crate::algorithms::model_free::q_learning::double_deep_q_learning_with_experience_replay::run_double_deep_q_learning_er;
use crate::algorithms::model_free::q_learning::double_deep_q_learning_with_prioritized_experience_replay::run_double_dqn_per;
use crate::algorithms::model_free::reinforce::reinforce::run_reinforce;
use crate::algorithms::model_free::reinforce::reinforce_mean_baseline::run_reinforce_baseline;
use crate::algorithms::model_free::reinforce::reinforce_baseline_learned_critic::run_reinforce_actor_critic;
use crate::algorithms::model_free::ppo::ppo_a2c::run_ppo_a2c;

// model-based
use crate::algorithms::model_based::monte_carlo::monte_carlo_random_rollout::run_random_rollout;
use crate::algorithms::model_based::monte_carlo::monte_carlo_tree_search_uct::run_mcts;
use crate::algorithms::model_based::expert_apprentice::expert_apprentice::expert_apprentice;

use crate::config::DeepLearningParams;

pub fn run_env_heuristic<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(env: &mut Env, env_name: &str, from_random: bool) {
    if !from_random {
        env.set_against_random();
    };
    let mut stdout = io::stdout();
    while !env.is_game_over() {
        reset_screen(&mut stdout, env_name);

        println!("{}", env);
        println!("Enter your action (or type 'quit' to exit): ");

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        let input = input.trim();
        if input.eq_ignore_ascii_case("quit") {
            println!("Exiting...");
            env.reset();
            break;
        }

        match input.parse::<usize>() {
            Ok(action) => {
                if env.available_actions().collect::<Vec<_>>().contains(&action) {
                    env.step(action);
                    if from_random {
                        env.switch_board();
                    }
                } else {
                    println!("Please enter a valid action");
                    let mut s = String::new();
                    io::stdin().read_line(&mut s).unwrap();
                }
            }
            Err(_) => {
                println!("Please enter a valid number or 'quit' to exit.");
                sleep(Duration::from_secs(1));
            }
        }
    }
    reset_screen(&mut stdout, "");
    println!("-------------------------------------");
    println!("Game Over!");
    println!("Score: {}", env.score());
    env.reset();
    end_of_run();
}

pub fn run_benchmark_random_agents<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(env: &mut Env, env_name: &str, _from_random: bool) {
    let mut stdout = io::stdout();
    reset_screen(&mut stdout, env_name);

    let mut time = 50;
    let mut input = String::new();
    println!("Enter the number of games to simulate:");
    io::stdin().read_line(&mut input).expect("Failed to read input");
    let num_games: usize = input.trim().parse().unwrap_or(10);

    input.clear();
    println!("Enable visual mode? (yes/no):");
    io::stdin().read_line(&mut input).expect("Failed to read input");
    let visual = input.trim().eq_ignore_ascii_case("yes");

    if visual {
        input.clear();
        println!("How long should a match last ? (in milliseconds) :");
        io::stdin().read_line(&mut input).expect("Failed to read input");
        time = input.trim().parse().unwrap_or(50);
    }
    let start = Instant::now();
    let mut games_played = 0;
    let mut rng = rand::thread_rng();
    let mut total_score = 0.0;

    if !env.set_against_random() { env.set_against_random(); }
    env.reset();

    for _ in 0..num_games {

        while !env.is_game_over() {
            let action_id = env.available_actions_ids().choose(&mut rng).unwrap();
            env.step_from_idx(action_id);
            if visual {
                let mut stdout = io::stdout();
                println!("{}", env);
                sleep(Duration::from_millis(time));
                reset_screen(&mut stdout, env_name);
            }
        }
        games_played += 1;
        total_score += env.score();
        env.reset();
    }

    let duration = start.elapsed();
    let games_per_second = games_played as f64 / duration.as_secs_f64();

    println!(
        "Joués: {} parties en {:?} secondes, avec un score de {:?} ({:.2} parties/sec)",
        games_played,
        duration.as_secs_f64(),
        total_score,
        games_per_second
    );
    end_of_run();
}

pub fn run_tests_all_algorithms<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(
    env_name: &str,
) {
    run_tests_model_free_algorithms::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name);
    run_tests_model_based_algorithms::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name);
}

/// Run only the **model-free** algorithms.
pub fn run_tests_model_free_algorithms<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(
    env_name: &str,
) {
    let mut params = DeepLearningParams::default();
    params.run_test = false;
    params.episode_stop = params.num_episodes / params.log_amount;

    for i in &params.group_testing {
        params.num_episodes = *i;
        params.episode_stop = i / params.log_amount;
        
        run_episodic_semi_gradient_sarsa::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        //run_tabular_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name, &params);
        run_deep_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        run_double_deep_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        run_double_deep_q_learning_er::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        run_double_dqn_per::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        run_reinforce::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        run_reinforce_baseline::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        run_reinforce_actor_critic::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        run_ppo_a2c::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
    }
}

/// Run only the **model-based** algorithms.
pub fn run_tests_model_based_algorithms<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(
    env_name: &str,
) {
    let mut params = DeepLearningParams::default();
    params.run_test = false;

    for i in &params.group_testing {
        params.num_episodes = *i;
        params.episode_stop = i / params.log_amount;
        
        run_random_rollout::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        run_mcts::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
        expert_apprentice::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params);
    }
}
