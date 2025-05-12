use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use std::fmt::Display;
use crate::config::{DeepLearningParams, MyBackend};
use crate::services::algorithms::model::{MyQmlp, Forward};
use crate::environments::env::DeepDiscreteActionsEnv;
use crate::gui::cli::common::end_of_run;
use crate::services::models::helpers::{compare_model_vs_random, compare_models};

fn csv_writer(path: &Path, header: &str) -> BufWriter<fs::File> {
    let new_file = !path.exists();
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("cannot create csv");
    let mut w = BufWriter::new(file);
    if new_file {
        w.write_all(header.as_bytes()).unwrap();
    }
    w
}

/// Helper: 0 → pick from env‐variable, otherwise keep the explicit value.
fn resolve_num_games(requested: usize, default_v: usize) -> usize {
    if requested != 0 {
        requested
    } else {
        std::env::var("NUM_GAMES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(default_v)
    }
}

pub fn run_compare_models<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>(
    env_name: &str,      
) where
    MyQmlp<MyBackend>: Forward<B = MyBackend>,
{
    let params = DeepLearningParams::default();
    let num_games = resolve_num_games(params.test_models_nb_games, 200);
    let env_root  = PathBuf::from("input").join(env_name);

    if !env_root.is_dir() {
        println!("No folder {:?} – nothing to compare.", env_root);

        end_of_run();
        return;
    }

    for crit_entry in fs::read_dir(&env_root).expect("cannot read env folder") {
        let crit_dir = crit_entry.expect("io error").path();
        if !crit_dir.is_dir() { continue; }

        let mut mpks: Vec<_> = fs::read_dir(&crit_dir)
            .unwrap()
            .filter_map(|e| {
                let p = e.unwrap().path();
                p.extension().map(|ext| ext == "mpk").unwrap_or(false).then_some(p)
            })
            .collect();
        mpks.sort();

        if mpks.len() < 2 {
            println!("criterion {:?} has < 2 models – skipping", crit_dir);
            continue;
        }

        let csv = crit_dir.join("results.csv");
        let mut wtr = csv_writer(
            &csv,
            "model_a,model_b,win_a,win_b\n",
        );

        for i in 0..mpks.len() - 1 {
            for j in i + 1..mpks.len() {
                let a = &mpks[i];
                let b = &mpks[j];

                // 1) A starts first
                let (wa1, wb1, _) = compare_models::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(
                    a.to_str().unwrap(),
                    b.to_str().unwrap(),
                    num_games,
                    params.num_tries,
                    env_name
                );
                // 2) B starts first
                let (wb2, wa2, _) = compare_models::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(
                    b.to_str().unwrap(),
                    a.to_str().unwrap(),
                    num_games,
                    params.num_tries,
                    env_name
                );

                // average the two runs
                let win_a = (wa1 + wa2) * 0.5;
                let win_b = (wb1 + wb2) * 0.5;

                writeln!(
                    wtr,
                    "{},{},{:.2},{:.2}",
                    a.file_name().unwrap().to_string_lossy(),
                    b.file_name().unwrap().to_string_lossy(),
                    win_a,
                    win_b
                )
                    .unwrap();

                println!(
                    "[{}] ⇄ [{}] → {:.1}% / {:.1}%",
                    a.file_name().unwrap().to_string_lossy(),
                    b.file_name().unwrap().to_string_lossy(),
                    win_a,
                    win_b
                );
            }
        }
    }

    end_of_run();
}

pub fn run_model_vs_random<
    const NUM_STATE_FEATURES: usize,
    const NUM_ACTIONS: usize,
    Env: DeepDiscreteActionsEnv<NUM_STATE_FEATURES, NUM_ACTIONS> + Display + Default + Clone,
>(
    env_name: &str,       
) where
    MyQmlp<MyBackend>: Forward<B = MyBackend>,
{
    let params = DeepLearningParams::default();
    let num_games = resolve_num_games(params.test_models_nb_games, 200);
    let env_root  = PathBuf::from("input").join(env_name);

    if !env_root.is_dir() {
        println!("No folder {:?} – nothing to test.", env_root);
        end_of_run();
        return;
    }

    for crit_entry in fs::read_dir(&env_root).expect("cannot read env folder") {
        let crit_dir = crit_entry.expect("io error").path();
        if !crit_dir.is_dir() { continue; }

        let csv = crit_dir.join("vs_random.csv");
        let mut wtr = csv_writer(
            &csv,
            "model,win,loss\n",
        );

        for m in fs::read_dir(&crit_dir).unwrap() {
            let mpk = m.unwrap().path();
            if mpk.extension().map(|e| e == "mpk").unwrap_or(false) {
                let (w, l, _) =
                    compare_model_vs_random::<NUM_STATE_FEATURES, NUM_ACTIONS, Env>(
                        mpk.to_str().unwrap(),
                        num_games,
                    );

                writeln!(
                    wtr,
                    "{},{:.2},{:.2}",
                    mpk.file_name().unwrap().to_string_lossy(), w, l
                ).unwrap();

                println!(
                    "[{}] vs random → {:.1}% / {:.1}%",
                    mpk.file_name().unwrap().to_string_lossy(),
                    w, l
                );
            }
        }
    }

    end_of_run();
}
