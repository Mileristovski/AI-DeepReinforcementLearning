use std::fmt::Display;
use std::io;
use crate::gui::cli::common::{reset_screen, user_choice};
use crate::services::envs::run_envs::{run_env_heuristic, run_benchmark_random_agents, run_tests_all_algorithms, run_tests_model_free_algorithms, run_tests_model_based_algorithms};
use crate::algorithms::model_free::sarsa::run_episodic_semi_gradient_sarsa;
use crossterm::terminal::disable_raw_mode;
use crate::algorithms::model_based::expert_apprentice::expert_apprentice::expert_apprentice;
use crate::algorithms::model_based::monte_carlo::monte_carlo_tree_search_uct::run_mcts;
use crate::algorithms::model_free::ppo::ppo_a2c::run_ppo_a2c;
use crate::algorithms::model_free::q_learning::deep_q_learning::run_deep_q_learning;
use crate::algorithms::model_free::q_learning::double_deep_q_learning::run_double_deep_q_learning;
use crate::algorithms::model_free::q_learning::double_deep_q_learning_with_experience_replay::run_double_deep_q_learning_er;
use crate::algorithms::model_free::q_learning::double_deep_q_learning_with_prioritized_experience_replay::run_double_dqn_per;
use crate::algorithms::model_free::q_learning::tabular_q_learning::run_tabular_q_learning;
use crate::algorithms::model_based::monte_carlo::monte_carlo_random_rollout::run_random_rollout;
use crate::algorithms::model_free::reinforce::reinforce::run_reinforce;
use crate::algorithms::model_free::reinforce::reinforce_baseline_learned_critic::run_reinforce_actor_critic;
use crate::algorithms::model_free::reinforce::reinforce_mean_baseline::run_reinforce_baseline;
use crate::config::DeepLearningParams;
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::services::models::helpers::play_vs_human;
use crate::services::models::runs::{run_compare_models, run_model_vs_random};

pub fn submenu_tests<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(env_name: &str)
{
    loop {
        let options = vec![
            "Test all algorithms",
            "Test model-free algorithms",
            "Test model-based algorithms",
            "Back",
        ];

        let message = format!("Testing menu for {}", env_name);
        let selected_index = user_choice(options.clone(), &message);
        let mut stdout = io::stdout();
        reset_screen(&mut stdout, env_name);

        match selected_index {
            0 => run_tests_all_algorithms::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name),
            1 => run_tests_model_free_algorithms::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name),
            2 => run_tests_model_based_algorithms::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name),
            3 => break,
            _ => {}
        }
    }
    disable_raw_mode().unwrap();
}

// ─────────────────────────────────────────────────────────────
//  PATCH: extend the first-level submenu to point to the new one
// ─────────────────────────────────────────────────────────────
pub fn submenu<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display,
>(env: &mut Env, env_name: &str)
{
    let mut from_random = true;

    loop {
        let options = vec![
            if from_random { "Random is on, turn random off" } else { "Random is off, Turn random on" },
            "Heuristic",
            "Train a model from the env",
            "Benchmark",
            "Group tests",
            "Test existing models vs each other",
            "Test existing models vs random",
            "Play vs a model",
            "Back",
        ];

        let message = format!("Menu for {}", env_name);
        let selected_index = user_choice(options.clone(), &message);
        let mut stdout = io::stdout();
        reset_screen(&mut stdout, env_name);

        match selected_index {
            0 => from_random = !from_random,
            1 => run_env_heuristic::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env, options[selected_index], from_random),
            2 => submenu_drl::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name),
            3 => run_benchmark_random_agents::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env, options[selected_index], from_random),
            4 => submenu_tests::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name),
            5 => run_compare_models::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name),
            6 => run_model_vs_random::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name),
            7 => play_vs_human::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(),
            8 => break,
            _ => {}
        }
    }
    disable_raw_mode().unwrap();
}

pub fn submenu_drl<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    const NUM_STATES: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(env_name: &str) {
    loop {
        let options = vec![
            "Semi gradient SARSA",
            "Tabular Q Learning",
            "Deep Q Learning",
            "Double Deep Q Learning",
            "Double Deep Q Learning With Experienced Replay",
            "Double Deep Q Learning With Prioritized Experienced Replay",
            "REINFORCE",
            "REINFORCE with mean baseline",
            "REINFORCE with Baseline Learned by a Critic ",
            "PPO A2C",
            "Monte Carlo Random Rollout",
            "Monte Carlo Tree Search",
            "Expert Apprentice",
            "Back"
        ];

        let message = format!("Training menu for {}", env_name);
        let selected_index = user_choice(options.clone(), &message);
        let mut stdout = io::stdout();
        reset_screen(&mut stdout, env_name);
        let params: DeepLearningParams = DeepLearningParams::default();

        match selected_index {
            0 => { run_episodic_semi_gradient_sarsa::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            1 => { run_tabular_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, NUM_STATES, Env>(env_name, &params); },
            2 => { run_deep_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            3 => { run_double_deep_q_learning::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name,&params); },
            4 => { run_double_deep_q_learning_er::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            5 => { run_double_dqn_per::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            6 => { run_reinforce::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            7 => { run_reinforce_baseline::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            8 => { run_reinforce_actor_critic::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            9 => { run_ppo_a2c::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            10 => { run_random_rollout::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            11 => { run_mcts::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            12 => { expert_apprentice::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(env_name, &params); },
            13 => { break; }
            _ => {}
        }
    }
    disable_raw_mode().unwrap();
}

