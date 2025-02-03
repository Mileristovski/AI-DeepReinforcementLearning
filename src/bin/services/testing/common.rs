use std::io;
use std::io::Stdout;
use std::process::Command;
use crossterm::event::{read, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{Clear, ClearType};

pub fn reset_screen(stdout: &mut Stdout, message: &str) {
    clear_screen();
    // Clear screen and draw the logo and menu
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

pub fn user_choice(options: Vec<&str>) -> usize {
    let mut stdout = io::stdout();
    let mut selected_index = 0;
    loop {
        reset_screen(&mut stdout, "Use Arrow Keys to Navigate, Enter to Select: \n");
        // Display menu options
        for (i, option) in options.iter().enumerate() {
            if i == selected_index {
                println!("> {}", option);
            } else {
                println!("  {}", option);
            }
        }

        let event = read().unwrap();
        if event == Event::Key(KeyCode::Up.into()) {
            if selected_index > 0 {
                selected_index -= 1;
            } else if selected_index == 0 {
                selected_index = options.len() - 1
            }

        } else if event == Event::Key(KeyCode::Down.into()) {
            if selected_index < options.len() - 1 {
                selected_index += 1;
            } else if selected_index == options.len() - 1 {
                selected_index = 0;
            }

        } else if event == Event::Key(KeyCode::Enter.into()) {
            return selected_index;
        } else if event == Event::Key(KeyCode::Esc.into()) {
            return selected_index;
        }
    }
}