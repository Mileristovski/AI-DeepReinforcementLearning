mod services;
mod environments;
mod algorithms;
mod config;
mod gui;

use crate::environments::env::DeepDiscreteActionsEnv;
use rand::prelude::IteratorRandom;
use rand::thread_rng;
use rand::prelude::*;
use crate::config::{MyAutodiffBackend, MyDevice};
use crate::environments::bobail::{BobailHeuristic, BB_NUM_ACTIONS, BB_NUM_STATE_FEATURES};
use crate::services::algorithms::helpers::{get_device, load_inference_model, step_with_model};
use crate::services::algorithms::model::MyQmlp;

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

fn print_board_state(env: &BobailHeuristic) {
    println!("\nBoard state:");
    println!("{}", env);

    println!("\nChip positions:");
    for (chip_val, pos) in &env.chip_lookup {
        println!("Chip {} at position {}", chip_val, pos);
    }

    println!("\nCurrent player: {}", env.current_player);
}


/*
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
}*/




fn full_dump(title: &str, env: &BobailHeuristic) {
    println!("\n================ {title} ================");
    println!("Current player : {}", if env.current_player == 1 { "Blue" } else { "Red" });
    println!("Bobail’s score : {}", env.score());
    println!("{}", env);
}

/*
fn main() {
    let mut env = BobailHeuristic::default();
    env.set_against_random();
    let mut rng = thread_rng();

    // --- initial position -------------------------------------------------
    full_dump("Initial position (Blue to move)", &env);

    // --- make one legal move ---------------------------------------------
    if let Some(action) = env.available_actions_ids().choose(&mut rng) {
        env.step_from_idx(action);
        full_dump("After Blue’s first move", &env);
    }

    // --- call switch_board() ---------------------------------------------
    env.switch_board();
    full_dump("After switch_board()", &env);

    // --- make another legal move in the mirrored position ----------------
    if let Some(action) = env.available_actions_ids().choose(&mut rng) {
        env.step_from_idx(action);
        full_dump("After Red’s reply (still mirrored)", &env);
    }

    // --- switch back again ------------------------------------------------
    env.switch_board();
    full_dump("After second switch_board()", &env);
}*/

type Env = BobailHeuristic;

fn main() {
    // ── 0. pick a device ────────────────────────────────────────────────
    let device: MyDevice = get_device();
    println!("Running on device: {:?}", device);

    // ── 1. create an *empty* template network (same constructor you use) ─
    let template = MyQmlp::<MyAutodiffBackend>::new(
        &device,
        BB_NUM_STATE_FEATURES,
        BB_NUM_ACTIONS,
    );

    // ── 2. load weights from disk and get the INNER (no-grad) model ─────
    let enemy = load_inference_model::<_, MyAutodiffBackend>(
        template,
        "./data/reinforce_lc/Bobail/run20250511_223831/reinforce_model_100.mpk",   // point to any existing checkpoint
        &device,
    );

    // ── 3. spin up an environment and play 100 half-moves via the helper ─
    let mut env = Env::default();
    env.set_against_random();        // red side will move randomly

    for ply in 0..100 {
        if env.is_game_over() {
            println!("Game ended after {ply} plies with score {}", env.score());
            break;
        }

        // agent (blue) chooses a *legal* action index…
        let mask = env.action_mask();
        let legal: Vec<_> = (0..BB_NUM_ACTIONS)
            .filter(|&i| mask[i] == 1.0)
            .collect();
        if legal.is_empty() {
            println!("No legal moves for the agent – terminating.");
            break;
        }
        let action_idx = *legal.iter().choose(&mut thread_rng()).unwrap();

        // …and we let `step_with_model` execute both sides of the turn
        step_with_model::<
            BB_NUM_STATE_FEATURES,
            BB_NUM_ACTIONS,
            _,
            MyAutodiffBackend,
            Env,
        >(&mut env, &enemy, action_idx, &device);
    }

    println!("Final board:\n{}", env);
    println!("Final score: {}", env.score());
}
