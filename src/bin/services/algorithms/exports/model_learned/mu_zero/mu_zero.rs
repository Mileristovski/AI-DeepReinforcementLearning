use std::fs::OpenOptions;
use serde::Serialize;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
use burn::module::Module;
use burn::tensor::backend::Backend;
use csv::Writer;
use crate::services::algorithms::exports::base_logger::{BaseLogger, RecordBase};

/// Flat tuple so csv::Writer can serialize it
#[derive(Serialize)]
pub struct MuZeroCsvRecord(
    usize,  // episode (iteration)
    f32,    // mean_score
    f64,    // total_elapsed_secs
    f64,    // interval_elapsed_secs
    usize,  // num_episodes
    usize,  // episode_stop
    usize,  // replay_capacity
    usize,  // batch_size
    usize,  // games_per_iter
    usize,  // mcts_sims
    f32,    // learning_rate
    f32,    // c (UCT constant)
);

pub struct MuZeroLogger {
    base: BaseLogger,
    num_episodes: usize,
    episode_stop: usize,
    replay_capacity: usize,
    batch_size: usize,
    games_per_iter: usize,
    mcts_sims: usize,
    learning_rate: f32,
    c: f32,
}

impl MuZeroLogger {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_dir: &str,
        params: &crate::config::DeepLearningParams
    ) -> Self {
        let base = BaseLogger::new(base_dir);
        MuZeroLogger {
            base,
            num_episodes:    params.num_episodes,
            episode_stop:    params.episode_stop,
            replay_capacity: params.mz_replay_cap,
            batch_size:      params.mz_batch_size,
            games_per_iter:  params.mz_games_per_iter,
            mcts_sims:       params.mcts_simulations,
            learning_rate:   params.ac_policy_lr,
            c:               params.mz_c,
        }
    }

    /// Log once per `episode_stop` episodes
    pub fn log(&mut self, episode: usize, mean_score: f32) {
        let base_metrics: RecordBase = self.base.make_base(episode, mean_score);
        let run_dir = self.base.run_dir().clone();

        println!(
            "MuZero Mean Score: {:.3} (ep {} — {:.2?} elapsed)",
            mean_score,
            episode,
            std::time::Duration::from_secs_f64(base_metrics.interval_elapsed_secs)
        );

        let rec = MuZeroCsvRecord(
            base_metrics.episode,
            base_metrics.mean_score,
            base_metrics.total_elapsed_secs,
            base_metrics.interval_elapsed_secs,
            self.num_episodes,
            self.episode_stop,
            self.replay_capacity,
            self.batch_size,
            self.games_per_iter,
            self.mcts_sims,
            self.learning_rate,
            self.c,
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
        let path = self.base.run_dir().join(format!("muzero_model_{episode}.mpk"));
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(path, &recorder)
            .expect("failed saving MuZero model");
    }
}
