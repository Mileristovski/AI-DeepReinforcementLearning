use crate::environments::env::Env;
use crate::environments::bobail::BobailEnv;
use crate::services::envs::common::reset_screen;
use crate::services::algo_helper::helpers::epsilon_greedy_action;
use crate::algorithms::sarsa::episodic_semi_gradient_sarsa;
use crate::environments::env::{DeepDiscreteActionsEnv, Forward};
use crate::config::*;
use crate::services::algo_helper::perceptron::{get_device, MyQmlp};
use std::fmt::Display;
use std::io;
use std::thread::sleep;
use std::time::Duration;
use rand::Rng;
use std::time::Instant;
use burn::module::AutodiffModule;
use burn::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn run_env_manually_solo<E: Env>(env: &mut E, env_name: &str, from_random: bool) {
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

pub fn benchmark_random_agents(env: &mut BobailEnv, env_name: &str, from_random: bool) {
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
    let mut rng = rand::thread_rng();
    let start = Instant::now();
    let mut games_played = 0;
    for _ in 0..num_games {
        if from_random { env.start_from_random_state() } else { env.reset()};

        while !env.is_game_over() {
            let available_actions: Vec<_> = env.available_actions();//.iter().cloned().collect();
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
        games_played += 1;
    }

    let duration = start.elapsed();
    let games_per_second = games_played as f64 / duration.as_secs_f64();

    println!(
        "Joués: {} parties en {:?} secondes, avec un score de {:?} ({:.2} parties/sec)",
        games_played,
        duration.as_secs_f64(),
        env.score(),
        games_per_second
    );
}

pub fn run_env_manually_random_1_v_1(env: &mut BobailEnv, env_name: &str, from_random: bool) {
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

pub fn run_deep_learning<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>()
{
    let device: MyDevice = get_device();
    println!("Using device: {:?}", device);

    // Create the model
    let model = MyQmlp::<MyAutodiffBackend>::new(&device,
                                                 NUM_STATE_FEATURES,
                                                 NUM_ACTIONS);


    let minus_one: Tensor<MyAutodiffBackend, 1> = Tensor::from_floats([-1.0; NUM_ACTIONS], &device);
    let plus_one: Tensor<MyAutodiffBackend, 1> = Tensor::from_floats([ 1.0; NUM_ACTIONS], &device);
    let fmin_vec: Tensor<MyAutodiffBackend, 1> = Tensor::from_floats([f32::MIN; NUM_ACTIONS], &device);

    // Train the model
    let model =
        episodic_semi_gradient_sarsa::<
            NUM_STATE_FEATURES,
            NUM_ACTIONS,
            _,
            MyAutodiffBackend,
            Env
        >(
            model,
            50_000,
            0.999f32,
            3e-3,
            1.0f32,
            1e-5f32,
            &minus_one,
            &plus_one,
            &fmin_vec,
            &device,
        );

    // Let's play some games (press enter to show the next game)
    let mut env = Env::default();
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let minus_one: Tensor<MyBackend, 1> = Tensor::from_floats([-1.0; NUM_ACTIONS], &device);
    let plus_one:  Tensor<MyBackend, 1> = Tensor::from_floats([ 1.0; NUM_ACTIONS], &device);
    let fmin_vec:  Tensor<MyBackend, 1> = Tensor::from_floats([f32::MIN; NUM_ACTIONS], &device);
    loop {
        env.reset();
        while !env.is_game_over() {
            println!("{}", env);
            let s = env.state_description();
            let s_tensor: Tensor<MyBackend, 1> = Tensor::from_floats(s.as_slice(), &device);

            let mask = env.action_mask();
            let mask_tensor: Tensor<MyBackend, 1> = Tensor::from(mask).to_device(&device);
            let q_s = model.valid().forward(s_tensor);

            let a = epsilon_greedy_action::<MyBackend, NUM_STATE_FEATURES, NUM_ACTIONS>(&q_s, &mask_tensor, &minus_one,  &plus_one,  &fmin_vec, env.available_actions_ids(), 1e-5f32, &mut rng, );
            env.step(a);
        }
        println!("{}", env);
        let mut s = String::new();
        io::stdin().read_line(&mut s).unwrap();
    }
}