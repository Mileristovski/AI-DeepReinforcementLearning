use std::fs::OpenOptions;
use serde::Serialize;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
use burn::module::Module;
use burn::tensor::backend::Backend;
use csv::Writer;
use crate::services::algorithms::exports::base_logger::{BaseLogger, RecordBase};

/// Flat tuple so csv::Writer can serialize it
#[derive(Serialize)]
pub struct AlphaZeroExpertCsvRecord(
    usize,  // iteration
    f32,    // mean_score
    f64,    // total_elapsed_secs
    f64,    // interval_elapsed_secs
    usize,  // num_iterations
    usize,  // episode_stop
    usize,  // games_per_iteration
    usize,  // mcts_simulations
    f32,    // c (UCT constant)
    f32,    // learning_rate
    f32,    // weight_decay
);

pub struct AlphaZeroExpertLogger {
    base: BaseLogger,
    num_iterations: usize,
    episode_stop: usize,
    games_per_iteration: usize,
    mcts_simulations: usize,
    c: f32,
    learning_rate: f32,
    weight_decay: f32,
}

impl AlphaZeroExpertLogger {
    pub fn new(base_dir: &str, params: &crate::config::DeepLearningParams) -> Self {
        let base = BaseLogger::new(base_dir);
        AlphaZeroExpertLogger {
            base,
            num_iterations:      params.num_episodes,
            episode_stop:        params.episode_stop,
            games_per_iteration: params.az_self_play_games,
            mcts_simulations:    params.mcts_simulations,
            c:                   params.az_c,
            learning_rate:       params.alpha,
            weight_decay:        params.opt_weight_decay_penalty,
        }
    }

    /// Log once per `episode_stop` iterations
    pub fn log(&mut self, episode: usize, mean_score: f32, mean_duration: std::time::Duration) {
        let base_metrics: RecordBase = self.base.make_base(episode, mean_score, mean_duration);
        let run_dir = self.base.run_dir().clone();

        println!(
            "Alpha-Zero-EA Mean Score: {:.3} / Mean Duration {:.3} (ep {} — {:.2?} elapsed)",
            mean_score,
            mean_duration.as_secs_f32(),
            episode,
            std::time::Duration::from_secs_f64(base_metrics.interval_elapsed_secs)
        );

        let rec = AlphaZeroExpertCsvRecord(
            base_metrics.episode,
            base_metrics.mean_score,
            base_metrics.total_elapsed_secs,
            base_metrics.interval_elapsed_secs,
            self.num_iterations,
            self.episode_stop,
            self.games_per_iteration,
            self.mcts_simulations,
            self.c,
            self.learning_rate,
            self.weight_decay,
        );

        let mut w = Writer::from_writer(
            OpenOptions::new()
                .append(true)
                .open(run_dir.join("metadata.csv")).unwrap(),
        );
        w.serialize(rec).unwrap();
        w.flush().unwrap();
    }

    /// Save model snapshot at given iteration
    pub fn save_model<M, B>(&self, model: &M, iter: usize)
    where
        M: Module<B>,
        B: Backend,
    {
        let path = self.base.run_dir().join(format!("alpha_zero_expert_model_{iter}.mpk"));
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(path, &recorder)
            .expect("failed saving AlphaZero-Expert model");
    }
}
