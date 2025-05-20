use std::io;
use std::io::Stdout;
use std::process::Command;
use crossterm::execute;
use crossterm::terminal::{Clear, ClearType};

pub fn reset_screen(stdout: &mut Stdout, message: &str) {
    clear_screen();
    
    execute!(stdout, Clear(ClearType::All)).unwrap();
    println!("----------------------------------------------------------------------");
    println!("{}", message);
    println!("----------------------------------------------------------------------");
}

pub fn end_of_run() {
    println!("\nPlease any key to exit...");
    io::stdin().read_line(&mut String::new()).unwrap();
}

pub fn clear_screen() {
    if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(&["/C", "cls"])
            .status()
            .unwrap();
    } else {
        Command::new("clear")
            .status()
            .unwrap();
    }
}

pub fn user_choice(options: Vec<&str>, message: &str) -> usize {
    use std::io::{self, Write};
    let mut stdout = io::stdout();
    let mut selected_index = String::new();

    reset_screen(&mut stdout, &*format!("Type the number of your selection, Enter to Select: \n{}", message));

    println!("Select an option:");
    for (i, option) in options.iter().enumerate() {
        println!("{}: {}", i + 1, option);
    }

    print!("Enter your choice: ");
    io::stdout().flush().unwrap();

    io::stdin().read_line(&mut selected_index).expect("Failed to read input");

    match selected_index.trim().parse::<usize>() {
        Ok(num) if num > 0 && num <= options.len() => num - 1,
        _ => {
            println!("Invalid input, defaulting to option 1.");
            0
        }
    }
}