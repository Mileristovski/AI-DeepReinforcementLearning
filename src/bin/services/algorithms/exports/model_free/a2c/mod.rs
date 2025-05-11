use std::fs::OpenOptions;
use serde::Serialize;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};
use burn::module::Module;
use burn::tensor::backend::Backend;
use csv::Writer;
use crate::services::algorithms::exports::base_logger::{BaseLogger, RecordBase};

/// Flat tuple so csv::Writer can serialize it
#[derive(Serialize)]
pub struct A2cCsvRecord(
    usize,  // episode
    f32,    // mean_score
    f64,    // total_elapsed_secs
    f64,    // interval_elapsed_secs
    usize,  // num_episodes
    usize,  // log_every
    usize,  // n_step
    f32,    // gamma
    f32,    // ent_coef
    f32,    // lr_pol
    f32,    // lr_val
    f32,    // weight_decay
);

pub struct A2cLogger {
    base: BaseLogger,
    num_episodes: usize,
    log_every: usize,
    n_step: usize,
    gamma: f32,
    ent_coef: f32,
    lr_pol: f32,
    lr_val: f32,
    weight_decay: f32,
}

impl A2cLogger {
    pub fn new(base_dir: &str, params: &crate::config::DeepLearningParams) -> Self {
        let base = BaseLogger::new(base_dir);
        A2cLogger {
            base,
            num_episodes: params.num_episodes,
            log_every:    params.episode_stop,
            n_step:       params.ac_n_step,
            gamma:        params.gamma,
            ent_coef:     params.ac_entropy_coef,
            lr_pol:       params.ac_policy_lr,
            lr_val:       params.ac_critic_lr,
            weight_decay: params.opt_weight_decay_penalty,
        }
    }

    /// Log once per `log_every` episodes
    pub fn log(&mut self, episode: usize, mean_score: f32, mean_duration: std::time::Duration) {
        let base: RecordBase = self.base.make_base(episode, mean_score, mean_duration);
        let run_dir = self.base.run_dir().clone();

        println!(
            "A2C Mean Score: {:.3} / Mean Duration {:.3} (ep {} — {:.2?} elapsed)",
            mean_score,
            mean_duration.as_secs_f32(),
            episode,
            std::time::Duration::from_secs_f64(base.interval_elapsed_secs)
        );

        let rec = A2cCsvRecord(
            base.episode,
            base.mean_score,
            base.total_elapsed_secs,
            base.interval_elapsed_secs,
            self.num_episodes,
            self.log_every,
            self.n_step,
            self.gamma,
            self.ent_coef,
            self.lr_pol,
            self.lr_val,
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

    /// Save policy model snapshot at given episode
    pub fn save_model<P, B>(&self, policy: &P, episode: usize)
    where
        P: Module<B>,
        B: Backend,
    {
        let path = self.base.run_dir().join(format!("a2c_policy_{episode}.mpk"));
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        policy.clone().save_file(path, &recorder).expect("failed saving A2C policy model");
    }
}
