use std::fs::OpenOptions;
use serde::Serialize;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
use burn::module::Module;
use burn::tensor::backend::Backend;
use csv::Writer;
use crate::services::algorithms::exports::base_logger::{BaseLogger, RecordBase};

/// Flat tuple so csv::Writer can serialize it
#[derive(Serialize)]
pub struct ReinforceCsvRecord(
    usize,  // episode
    f32,    // mean_score
    f64,    // total_elapsed_secs
    f64,    // interval_elapsed_secs
    usize,  // num_episodes
    usize,  // log_every
    f32,    // gamma
    f32,    // lr
    f32,    // weight_decay
);

pub struct ReinforceLogger {
    base: BaseLogger,
    num_episodes: usize,
    log_every: usize,
    gamma: f32,
    lr: f32,
    weight_decay: f32,
}

impl ReinforceLogger {
    pub fn new(base_dir: &str, params: &crate::config::DeepLearningParams) -> Self {
        let base = BaseLogger::new(base_dir);
        ReinforceLogger {
            base,
            num_episodes: params.num_episodes,
            log_every:    params.episode_stop,
            gamma:        params.gamma,
            lr:           params.alpha,
            weight_decay: params.opt_weight_decay_penalty,
        }
    }

    /// Log once per `log_every` episodes
    pub fn log(&mut self, episode: usize, mean_score: f32) {
        let base: RecordBase = self.base.make_base(episode, mean_score);
        let run_dir = self.base.run_dir().clone();

        println!(
            "REINFORCE Mean Score: {:.3} (ep {} — {:.2?} elapsed)",
            mean_score,
            episode,
            std::time::Duration::from_secs_f64(base.interval_elapsed_secs)
        );

        let rec = ReinforceCsvRecord(
            base.episode,
            base.mean_score,
            base.total_elapsed_secs,
            base.interval_elapsed_secs,
            self.num_episodes,
            self.log_every,
            self.gamma,
            self.lr,
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

    /// Save model snapshot at given episode
    pub fn save_model<M, B>(&self, model: &M, episode: usize)
    where
        M: Module<B>,
        B: Backend,
    {
        let path = self.base.run_dir().join(format!("reinforce_model_{episode}.mpk"));
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(path, &recorder).expect("failed saving REINFORCE model");
    }
}
