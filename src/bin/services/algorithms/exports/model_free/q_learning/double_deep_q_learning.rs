use std::fs::OpenOptions;
use serde::Serialize;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
use burn::module::Module;
use burn::tensor::backend::Backend;
use csv::Writer;
use crate::services::algorithms::exports::base_logger::{BaseLogger, RecordBase};

/// Flat tuple so csv::Writer can serialize it
#[derive(Serialize)]
pub struct DoubleDqnCsvRecord(
    usize,   // episode
    f32,     // mean_score
    f32,     // mean_duration
    f64,     // total_elapsed_secs
    f64,     // interval_elapsed_secs
    usize,   // num_episodes
    usize,   // episode_stop
    f32,     // gamma
    f32,     // alpha
    f32,     // start_epsilon
    f32,     // final_epsilon
    f32,     // weight_decay
);

pub struct DoubleDqnLogger {
    base: BaseLogger,
    num_episodes: usize,
    episode_stop: usize,
    gamma: f32,
    alpha: f32,
    start_epsilon: f32,
    final_epsilon: f32,
    weight_decay: f32,
}

impl DoubleDqnLogger {
    pub fn new(base_dir: &str, env_name: String, params: &crate::config::DeepLearningParams) -> Self {
        let base = BaseLogger::new(base_dir, env_name);
        DoubleDqnLogger {
            base,
            num_episodes:    params.num_episodes,
            episode_stop:    params.episode_stop,
            gamma:           params.gamma,
            alpha:           params.alpha,
            start_epsilon:   params.start_epsilon,
            final_epsilon:   params.final_epsilon,
            weight_decay:    params.opt_weight_decay_penalty,
        }
    }

    /// log at each `episode_stop` block
    pub fn log(&mut self, episode: usize, mean_score: f32, mean_duration: std::time::Duration) {
        let base: RecordBase = self.base.make_base(episode, mean_score, mean_duration);
        let run_dir = self.base.run_dir().clone();

        println!(
            "DDQL Mean Score: {:.3} / Mean Duration {:.3} (ep {} — {:.2?} elapsed)",
            mean_score,
            mean_duration.as_secs_f32(),
            episode,
            std::time::Duration::from_secs_f64(base.interval_elapsed_secs)
        );

        let rec = DoubleDqnCsvRecord(
            base.episode,
            base.mean_score,
            base.mean_duration.as_secs_f32(),
            base.total_elapsed_secs,
            base.interval_elapsed_secs,
            self.num_episodes,
            self.episode_stop,
            self.gamma,
            self.alpha,
            self.start_epsilon,
            self.final_epsilon,
            self.weight_decay,
        );

        let mut w = Writer::from_writer(
            OpenOptions::new()
                .append(true)
                .open(run_dir.join("metadata.csv"))
                .unwrap(),
        );
        w.serialize(rec).unwrap();
        w.flush().unwrap();
    }

    /// save model snapshot
    pub fn save_model<M, B>(&self, model: &M, ep: usize)
    where
        M: Module<B>,
        B: Backend,
    {
        let path = self.base.run_dir().join(format!("double_dql_model_{ep}.mpk"));
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(path, &recorder).expect("failed saving Double DQN model");

        let input_dir = self.base.run_input_dir();
        let input_path = input_dir.join(format!("double_dql_model_{ep}.mpk"));
        let rec  = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(input_path, &rec).expect("failed saving model");

    }
}
