use std::fs::OpenOptions;
use serde::Serialize;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
use burn::module::Module;
use burn::tensor::backend::Backend;
use csv::Writer;
use crate::services::algorithms::exports::base_logger::{BaseLogger, RecordBase};

/// Flat tuple so csv::Writer can serialize it
#[derive(Serialize)]
pub struct DqnCsvRecord(
    usize,   // episode
    f32,     // mean_score
    f64,     // total_elapsed_secs
    f64,     // interval_elapsed_secs
    usize,   // num_episodes
    usize,   // episode_stop
    f32,     // gamma
    f32,     // lr
    f32,     // eps_start
    f32,     // eps_final
);

pub struct DqnLogger {
    base: BaseLogger,
    num_episodes: usize,
    episode_stop: usize,
    gamma: f32,
    lr: f32,
    eps_start: f32,
    eps_final: f32,
}

impl DqnLogger {
    pub fn new(base_dir: &str, params: &crate::config::DeepLearningParams) -> Self {
        let base = BaseLogger::new(base_dir);
        DqnLogger {
            base,
            num_episodes:    params.num_episodes,
            episode_stop:    params.episode_stop,
            gamma:           params.gamma,
            lr:              params.alpha,          // your code uses `alpha` as lr
            eps_start:       params.start_epsilon,
            eps_final:       params.final_epsilon,
        }
    }

    /// log at each `episode_stop` block
    pub fn log(&mut self, episode: usize, mean_score: f32) {
        // get timing info
        let base: RecordBase = self.base.make_base(episode, mean_score);
        let run_dir = self.base.run_dir().clone();

        // print to console
        println!(
            "DQN Mean Score: {:.3} (ep {} — {:.2?} elapsed)",
            mean_score, episode, std::time::Duration::from_secs_f64(base.interval_elapsed_secs)
        );

        // build flat record
        let rec = DqnCsvRecord(
            base.episode,
            base.mean_score,
            base.total_elapsed_secs,
            base.interval_elapsed_secs,
            self.num_episodes,
            self.episode_stop,
            self.gamma,
            self.lr,
            self.eps_start,
            self.eps_final,
        );

        // append
        let mut w = Writer::from_writer(
            OpenOptions::new()
                .append(true)
                .open(run_dir.join("metadata.csv"))
                .unwrap(),
        );
        w.serialize(rec).unwrap();
        w.flush().unwrap();
    }

    /// save your online model at chosen episodes
    pub fn save_model<M, B>(&self, model: &M, ep: usize)
    where
        M: Module<B>,
        B: Backend,
    {
        let path = self.base.run_dir().join(format!("dqn_model_{ep}.mpk"));
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(path, &recorder).expect("failed saving DQN model");
    }
}
