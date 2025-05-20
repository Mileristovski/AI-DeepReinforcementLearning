use std::fs::OpenOptions;
use serde::Serialize;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
use burn::module::Module;
use burn::tensor::backend::Backend;
use csv::Writer;
use crate::services::algorithms::exports::base_logger::{BaseLogger, RecordBase};

/// Flat tuple so csv::Writer can serialize it
#[derive(Serialize)]
pub struct ReinforceBaselineCsvRecord(
    usize,  // episode
    f32,    // mean_score
    f32,    // mean_duration
    f64,    // total_elapsed_secs
    f64,    // interval_elapsed_secs
    usize,  // num_episodes
    usize,  // episode_stop
    f32,    // gamma
    f32,    // lr
);

pub struct ReinforceBaselineLogger {
    base: BaseLogger,
    num_episodes: usize,
    episode_stop: usize,
    gamma: f32,
    lr: f32,
}

impl ReinforceBaselineLogger {
    pub fn new(base_dir: &str, env_name: String, params: &crate::config::DeepLearningParams) -> Self {
        let base = BaseLogger::new(base_dir, env_name);
        ReinforceBaselineLogger {
            base,
            num_episodes: params.num_episodes,
            episode_stop: params.episode_stop,
            gamma: params.gamma,
            lr: params.alpha,
        }
    }

    /// Log once per `episode_stop` episodes
    pub fn log(&mut self, episode: usize, mean_score: f32, mean_duration: std::time::Duration) {
        let base_metrics: RecordBase = self.base.make_base(episode, mean_score, mean_duration);
        let run_dir = self.base.run_dir().clone();

        println!(
            "REINFORCE-BASE Mean Score: {:.3} / Mean Duration {:.3} (ep {} — {:.2?} elapsed)",
            mean_score,
            mean_duration.as_secs_f32(),
            episode,
            std::time::Duration::from_secs_f64(base_metrics.interval_elapsed_secs)
        );

        let rec = ReinforceBaselineCsvRecord(
            base_metrics.episode,
            base_metrics.mean_score,
            base_metrics.mean_duration.as_secs_f32(),
            base_metrics.total_elapsed_secs,
            base_metrics.interval_elapsed_secs,
            self.num_episodes,
            self.episode_stop,
            self.gamma,
            self.lr,
        );

        let mut w = Writer::from_writer(
            OpenOptions::new()
                .append(true)
                .open(run_dir.join("metadata.csv")).unwrap(),
        );
        w.serialize(rec).unwrap();
        w.flush().unwrap();
    }

    /// Save model snapshot at given episode
    pub fn save_model<M, B>(&self, model: &M, episode: usize)
    where
        M: Module<B>,
        B: Backend,
    {
        let path = self.base.run_dir().join(format!("reinforce_base_model_{episode}.mpk"));
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(path, &recorder).expect("failed saving REINFORCE-BASE model");

        let input_dir = self.base.run_input_dir();
        let input_path = input_dir.join(format!("reinforce_base_model_{episode}.mpk"));
        let rec  = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(input_path, &rec).expect("failed saving model");

    }
}
