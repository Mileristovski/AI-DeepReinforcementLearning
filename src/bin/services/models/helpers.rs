use std::fmt::Display;
use std::io;
use std::thread::sleep;
use std::time::{Duration, Instant};
use burn::prelude::Tensor;
use crate::config::{MyAutodiffBackend, MyBackend, MyDevice};
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::gui::cli::common::reset_screen;
use crate::services::algorithms::helpers::{get_device, greedy_policy_action, load_inference_model};
use crate::services::algorithms::model::{Forward, MyQmlp};

pub fn compare_models<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>(
    model_path_1: &str,
    model_path_2: &str,
    num_games: usize,
    num_tries: usize,
    env_name: &str,
    visual: bool,
    time: u64
) -> (f32, f32, f32, f64, f32) 
where
    MyQmlp<MyBackend>: Forward<B = MyBackend>,
{
    let model1_name = std::path::Path::new(model_path_1)
        .file_name()
        .unwrap()
        .to_string_lossy();
    let model2_name = std::path::Path::new(model_path_2)
        .file_name()
        .unwrap()
        .to_string_lossy();
    
    // 0. pick a device 
    let device: MyDevice = get_device();

    // 1. create an *empty* template network (same constructor you use) 
    let template = MyQmlp::<MyAutodiffBackend>::new(
        &device,
        NUM_STATE_FEATURES,
        NUM_ACTIONS,
    );

    // 2. load weights from disk and get the INNER (no-grad) model 
    let valid_model1 = load_inference_model::<_, MyAutodiffBackend>(
        template.clone(),
        model_path_1,   // point to any existing checkpoint
        &device,
    );
    
    let valid_model2 = load_inference_model::<_, MyAutodiffBackend>(
        template,
        model_path_2,   // point to any existing checkpoint
        &device,
    );
    
    let fmin_vec = Tensor::<MyBackend, 1>::from_floats([f32::MIN; NUM_ACTIONS], &device);

    let mut model1_wins = 0f32;
    let mut model2_wins = 0f32;
    let mut draws = 0f32;
    
    println!("Starting tests {} vs {}...", model1_name, model2_name);
    let mut all_games_duration: Vec<Duration> = Vec::new();
    let mut total_steps = 0usize;
    let mut avrg_step = 0.0;
    
    for game in 0..num_games {
        let mut env = Env::default();
        let (first_model, second_model) = if game % 2 == 0 {
            (&valid_model1, &valid_model2)
        } else {
            (&valid_model2, &valid_model1)
        };
        let mut current_player = 0;
        let mut bobail: bool = false;
        
        env.set_against_random();
        let mut count = 0usize;

        let game_start = Instant::now();
        while !env.is_game_over() && count < num_tries {
            if visual {
                let mut stdout = io::stdout();
                println!("{}", env);
                sleep(Duration::from_millis(time));
                reset_screen(&mut stdout, env_name);
            }
            
            let current_model = if current_player == 0 { first_model } else { second_model };

            let s_tensor = Tensor::<MyBackend, 1>::from_floats(env.state_description().as_slice(), &device);
            let mask_tensor = Tensor::<MyBackend, 1>::from(env.action_mask()).to_device(&device);

            let out = current_model.forward(s_tensor);
            let policy_logits = out.clone().slice([0..NUM_ACTIONS]);
            let action = greedy_policy_action::<MyBackend>(&policy_logits, &mask_tensor, &fmin_vec);
            
            env.switch_board();
            env.step_from_idx(action);
            env.switch_board();
            if !bobail {
                bobail = !bobail;
            } else {
                current_player = if current_player == 1 { 0 } else  { 1 };
                bobail = !bobail;
            }
            count += 1;
        }
        all_games_duration.push(game_start.elapsed());
        total_steps += count;
        avrg_step += count as f32;
        
        let final_score = env.score();
        if final_score == 0.0 {
            draws += 1.0;
        } else if game % 2 == 0 {
            if final_score > 0.0 { model1_wins += 1.0; } else { model2_wins += 1.0; }
        } else {
            if final_score > 0.0 { model2_wins += 1.0; } else { model1_wins += 1.0; }
        }

        // Print progress
        if (game + 1) % 200 == 0 {
            println!("Completed {}/{} games →  current stats: {} wins: {:.1}%, {} wins: {:.1}%, Draws: {:.1}%, Mean_duration: {}s, Mean_steps {:.1} steps",
                     game + 1,
                     num_games,
                     model1_name,
                     100.0 * model1_wins / (game as f32 + 1.0),
                     model2_name,
                     100.0 * model2_wins / (game as f32 + 1.0),
                     100.0 * draws / (game as f32 + 1.0),
                     game_start.elapsed().as_secs_f64(),
                     avrg_step / 200.0,);
            avrg_step = 0.0;
        }
    }

    let mean_duration = all_games_duration.iter().sum::<Duration>() / all_games_duration.len() as u32;
    // Convert to percentages
    let total_games = num_games as f32;
    (
        100.0 * model1_wins / total_games,
        100.0 * model2_wins / total_games,
        100.0 * draws / total_games,
        mean_duration.as_secs_f64(),
        (total_steps as f32) / (num_games as f32)
    )
}


pub fn compare_model_vs_random<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env,
>(
    model_path: &str,
    num_games:  usize,
    num_tries: usize,
) -> (f32, f32, f32, f64, f32)        
where
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS>
    + Display
    + Default
    + Clone,
    MyQmlp<MyBackend>: Forward<B = MyBackend>,
{
    let model_name = std::path::Path::new(model_path)
        .file_name()
        .unwrap()
        .to_string_lossy();
    // 0. pick device 
    let device: MyDevice = get_device();

    // 1. template net + load weights (no-grad inner model) 
    let template = MyQmlp::<MyAutodiffBackend>::new(&device,
                                                    NUM_STATE_FEATURES,
                                                    NUM_ACTIONS);
    let policy = load_inference_model::<_, MyAutodiffBackend>(
        template,
        model_path,
        &device,
    );

    // 2. bookkeeping 
    let fmin_vec = Tensor::<MyBackend, 1>::from_floats([f32::MIN; NUM_ACTIONS],
                                                       &device);
    let mut model_wins  = 0f32;
    let mut random_wins = 0f32;
    let mut draws       = 0f32;
    let mut count = 0usize;

    println!("Starting tests {} vs random...", model_name);
    // 3. play many games 
    let mut env = Env::default();
    let mut all_games_duration: Vec<Duration> = Vec::new();
    let mut total_steps = 0usize;
    let mut avrg_steps = 0.0;
    
    for game in 0..num_games {
        count = 0;
        env.reset();
        let game_start = Instant::now();
        
        while !env.is_game_over() && count < num_tries {
            let mask = env.action_mask();
            let s_tensor = Tensor::<MyBackend, 1>::from_floats(
                env.state_description().as_slice(),
                &device,
            );
            let mask_t = Tensor::<MyBackend, 1>::from(mask).to_device(&device);

            let logits   = policy.forward(s_tensor);
            let logits   = logits.slice([0..NUM_ACTIONS]);
            let action   = greedy_policy_action::<MyBackend>(&logits,
                                                             &mask_t,
                                                             &fmin_vec);
            env.step_from_idx(action);
            count += 1;
        }
        all_games_duration.push(game_start.elapsed());
        total_steps += count;
        avrg_steps += count as f32;

        // 4. update statistics 
        if count >= num_tries {
            draws += 1.0;
        } else {
            if env.score() == 1.0 {
                model_wins += 1.0
            } else {
                random_wins += 1.0
            }
        }

        if (game + 1) % 200 == 0 {
            println!(
                "Completed {}/{} games → {} {:.1} %, random {:.1} %, draws {:.1} %, mean_duration {}s, mean_steps {:.1} steps",
                game + 1,
                num_games,
                model_name,
                100.0 * model_wins  / (game + 1) as f32,
                100.0 * random_wins / (game + 1) as f32,
                100.0 * draws       / (game + 1) as f32,
                game_start.elapsed().as_secs_f64(),
                avrg_steps / 200.0,
            );
            avrg_steps = 0.0;
        }
    }

    let mean_duration = all_games_duration.iter().sum::<Duration>() / all_games_duration.len() as u32;
    // 5. percentages 
    let n = num_games as f32;
    (
        100.0 * model_wins  / n,
        100.0 * random_wins / n,
        100.0 * draws       / n,
        mean_duration.as_secs_f64(),
        (total_steps as f32) / (num_games as f32)
    )
}
