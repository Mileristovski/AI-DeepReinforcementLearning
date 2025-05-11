use crate::environments::env::DeepDiscreteActionsEnv;
use crate::gui::cli::common::{end_of_run, reset_screen};
use std::fmt::Display;
use std::io;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;
use rand::prelude::IteratorRandom;

pub fn run_env_heuristic<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(env: &mut Env, env_name: &str, from_random: bool) {
    if from_random {
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
            break;
        }

        match input.parse::<usize>() {
            Ok(action) => {
                if env.available_actions().collect::<Vec<_>>().contains(&action) {
                    env.step(action);
                    env.switch_board();
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
>(env: &mut Env, env_name: &str, from_random: bool) {
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
    for _ in 0..num_games {
        if from_random { env.set_from_random_state() };

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
