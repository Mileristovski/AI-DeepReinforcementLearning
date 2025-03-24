use std::io;
use std::thread::sleep;
use std::time::Duration;
use rand::Rng;
use crate::environments::env::Env;
use crate::environments::bobail::BobailEnv;
use crate::services::testing::common::reset_screen;
use std::time::Instant;

pub fn testing_env_manually<E: Env>(env: &mut E, env_name: &str, from_random: bool) {
    if from_random { env.start_from_random_state() };

    let mut stdout = io::stdout();
    while !env.is_game_over() {
        reset_screen(&mut stdout, env_name);

        env.display();
        println!("Score: {}", env.score());

        let available_actions: Vec<_> = env.available_actions().iter().cloned().collect();
        println!("Available actions: {:?}", available_actions);
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

        match input.parse::<i32>() {
            Ok(action) => {
                if available_actions.contains(&action) {
                    env.step(action);
                } else {
                    println!("Invalid action: {}", action);
                    sleep(Duration::from_secs(1));
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
}

pub fn testing_env_manually_random(env: &mut BobailEnv, env_name: &str, from_random: bool) {
    if from_random { env.start_from_random_state() };

    let mut stdout = io::stdout();
    while !env.is_game_over() {
        if env.current_player == env.red_player || (env.current_player == env.bobail && env.previous_player == env.blue_player) {
            let available_actions: Vec<_> = env.available_actions().iter().cloned().collect();
            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..available_actions.len());

            let action = Some(available_actions[index]);
            env.step(action.unwrap());
        } else {
            reset_screen(&mut stdout, env_name);

            env.display();
            println!("Score: {}", env.score());

            let available_actions: Vec<_> = env.available_actions().iter().cloned().collect();
            println!("Available actions: {:?}", available_actions);
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

            match input.parse::<i32>() {
                Ok(action) => {
                    if available_actions.contains(&action) {
                        env.step(action);
                    } else {
                        println!("Invalid action: {}", action);
                        sleep(Duration::from_secs(1));
                    }
                }
                Err(_) => {
                    println!("Please enter a valid number or 'quit' to exit.");
                    sleep(Duration::from_secs(1));
                }
            }
        }
    }
    reset_screen(&mut stdout, "");
    println!("-------------------------------------");
    println!("Game Over!");
    println!("Score: {}", env.score());
    env.reset();
}

pub fn benchmark_random_agents(env: &mut BobailEnv, env_name: &str, from_random: bool) {
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
    let mut rng = rand::thread_rng();
    let start = Instant::now();
    let mut games_played = 0;
    let mut score_total = 0.0;
    for _ in 0..num_games {
        if from_random { env.start_from_random_state() } else { env.reset()};

        while !env.is_game_over() {
            let available_actions: Vec<_> = env.available_actions().iter().cloned().collect();
            if available_actions.is_empty() {
                break;
            }
            let index = rng.gen_range(0..available_actions.len());
            let action = available_actions[index];
            env.step(action);
            if visual {
                let mut stdout = io::stdout();
                env.display();
                sleep(Duration::from_millis(time));
                reset_screen(&mut stdout, env_name);
            }
        }
        score_total += env.score();
        games_played += 1;
    }

    let duration = start.elapsed();
    let games_per_second = games_played as f64 / duration.as_secs_f64();

    println!(
        "Jouées: {} parties en {:?} secondes, avec un score de {:?} ({:.2} parties/sec)",
        games_played,
        duration.as_secs_f64(),
        score_total,
        games_per_second
    );
}