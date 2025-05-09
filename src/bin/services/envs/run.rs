use crate::environments::env::DeepDiscreteActionsEnv;
use crate::gui::cli::common::reset_screen;
use crate::services::algo_helper::helpers::{epsilon_greedy_action, get_device};
use crate::algorithms::sarsa::episodic_semi_gradient_sarsa;
use crate::config::*;
use crate::services::algo_helper::qmlp::{Forward, MyQmlp};
use std::fmt::Display;
use std::io;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;
use burn::module::AutodiffModule;
use burn::prelude::*;
use rand::prelude::IteratorRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

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
                if env.available_actions_ids().collect::<Vec<_>>().contains(&action) {
                    env.step(action);
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
}

pub fn benchmark_random_agents<    
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display
>(env: &mut Env, env_name: &str/*, from_random: bool*/) {
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
    for _ in 0..num_games {
        // if from_random { env.start_from_random_state() } else { env.reset()};

        while !env.is_game_over() {
            let action = env.available_actions_ids().choose(&mut rng).unwrap();
            env.step(action);
            if visual {
                let mut stdout = io::stdout();
                println!("{}", env);
                sleep(Duration::from_millis(time));
                reset_screen(&mut stdout, env_name);
            }
        }
        games_played += 1;
        env.reset();
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
    let parameters = DeepLearningParams::default();

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
            parameters.num_episodes,
            parameters.gamma,
            parameters.alpha,
            parameters.start_epsilon,
            parameters.final_epsilon,
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

            let a = epsilon_greedy_action::<MyBackend, NUM_STATE_FEATURES, NUM_ACTIONS>(&q_s, &mask_tensor, &minus_one,  &plus_one,  &fmin_vec, env.available_actions_ids(), 1e-5f32, &mut rng);
            env.step(a);
        }
        println!("{}", env);
        let mut s = String::new();
        io::stdin().read_line(&mut s).unwrap();
    }
}