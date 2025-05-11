mod services;
mod environments;
mod algorithms;
mod config;
mod gui;

use crate::environments::env::DeepDiscreteActionsEnv;
use crate::environments::bobail::BobailHeuristic;
use rand::prelude::IteratorRandom;

type GameEnv = BobailHeuristic;

fn print_state_description_chunks(env: &impl DeepDiscreteActionsEnv<300, 48>) {
    let state = env.state_description();
    println!("=== STATE DESCRIPTION (chunked by cell) ===");

    for i in 0..25 {
        let chunk = &state[i * 12..(i + 1) * 12];
        print!("Cell {:>2}: [", i);
        for (j, val) in chunk.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{:.1}", val);
        }
        println!("]");
    }
}


fn display_properties(env: &mut GameEnv) {
    println!("\n=== STATE DESCRIPTION ===");
    println!("{:?}", env.state_description());
    print_state_description_chunks(env);

    println!("\n=== BOARD ===");
    println!("{:?}", env.board);

    let available_ids = env.available_actions_ids().collect::<Vec<_>>();
    let mask = env.action_mask();

    println!("\n=== AVAILABLE ACTION IDS ===");
    println!("{:?}", available_ids);

    println!("\n=== ACTION MASK ===");
    println!("{:?}", mask);
    for (i, &m) in mask.iter().enumerate() {
        println!("Action {:>2}: Mask = {}", i, m);
    }

    println!("\n=== ACTIONS ENABLED IN MASK ===");
    let mask_actions: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v > 0.0 { Some(i) } else { None })
        .collect();
    println!("{:?}", mask_actions);

    println!("\n=== COMPARISON ===");
    if mask_actions == available_ids {
        println!("✅ Mask matches available action IDs.");
    } else {
        println!("❌ Mismatch! Mask and available action IDs differ.");
        println!("Expected (from mask): {:?}", mask_actions);
        println!("Returned (from available_actions_ids): {:?}", available_ids);
    }

    println!("\n=== ONE-HOT CHECK ===");
    for i in 0..env.board.len() {
        let expected = env.board[i] as usize;
        let desc_slice = &env.state_description()[i * 12..(i + 1) * 12];
        let is_one_hot = desc_slice.iter().filter(|&&v| v == 1.0).count() == 1;
        let correct_index = desc_slice[expected] == 1.0;
        println!(
            "Cell {:>2}: board = {:>2}, one-hot = {}, correct pos = {}",
            i, expected, is_one_hot, correct_index
        );
    }
}

fn main() {
    let mut env = GameEnv::default();
    env.reset();
    // env.set_against_random();

    println!("\n=== Initial State ===");
    display_properties(&mut env);

    let mut env = BobailHeuristic::default();
    for _ in 0..10_000 {
        let mask = env.action_mask();
        let legal: Vec<_> = (0..48).filter(|&i| mask[i] == 1.0).collect();
        if legal.is_empty() { break; }
        let a = *legal.iter().choose(&mut rand::thread_rng()).unwrap();
        env.step_from_idx(a);          // should never panic
    }
    println!("Random play survived without panics ✅");
}
/*
    for i in 0..10 {
        if env.is_game_over() {
            println!("\nGame over reached early at step {}", i);
            break;
        }

        if let Some(action) = env.available_actions_ids().choose(&mut rand::thread_rng()) {
            println!("\n>>> Step {}: Executing action {} <<<", i + 1, action);
            env.step_from_idx(action);
            if env.current_player == 1 || env.previous_player != 1 {
                display_properties(&mut env);
            }
            sleep(Duration::from_millis(500));
        } else {
            println!("\nNo actions available at step {}", i);
            break;
        }
    }

    println!("\n=== Final State ===");
    println!("{}", env);
}
*/
// fn main() {
    // let mut env = GameEnv::default();
    // // env.is_random_state = true;
    // env.reset();
    // 
    // println!("{}", env);
    // display_properties(&mut env);
    // env.step(1);
    // display_properties(&mut env);
    // println!("{}", env);
    // 
    // env.step(0);
    // display_properties(&mut env);
    // println!("{}", env);
    //     
    // env.step(3);
    // display_properties(&mut env);
    // println!("{}", env);
// 
    // env.step(3);
    // display_properties(&mut env);
    // println!("{}", env);
    // 
    // display_properties(&mut env);
    // unimplemented!("This should be used as a test script")
// }
