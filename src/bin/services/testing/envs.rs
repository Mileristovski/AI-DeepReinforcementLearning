use std::io;
use std::thread::sleep;
use std::time::Duration;
use crate::environments::env::Env;
use crate::services::testing::common::reset_screen;

pub fn testing_env_manually<E: Env>(env: &mut E, env_name: &str) {
    env.from_random_state();
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