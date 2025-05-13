use std::fs::{create_dir_all, metadata, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;
use csv::Writer;
use chrono::Local;

/// A generic record type: any algorithm can extend this by embedding their own extra fields.
#[derive(Clone)]               
pub struct RecordBase {
    pub episode: usize,
    pub mean_score: f32,
    pub mean_duration: std::time::Duration,
    pub total_elapsed_secs: f64,
    pub interval_elapsed_secs: f64,
}

/// BaseLogger handles directory creation, CSV setup, and writing base records.
#[allow(dead_code)]
pub struct BaseLogger {
    writer: Writer<std::fs::File>,
    start_time: Instant,
    pub(crate) last_log_time: Instant,
    run_dir: PathBuf,
    input_dir: PathBuf,
    env_name: String
}

impl BaseLogger {
    /// Create a new run folder under `base_dir`, open CSV with `metadata.csv`.
    pub fn new(base_dir: &str, env_name: String) -> Self {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let run_dir = Path::new(base_dir).join(format!("run{timestamp}"));
        let path = format!("./input/{}", env_name);
        let input_dir = Path::new(&path).join("evaluation".to_string());
        create_dir_all(&run_dir).expect("could not create run directory");
        create_dir_all(&input_dir).expect("could not create run directory");

        let csv_path = run_dir.join("metadata.csv");
        let is_new = !csv_path.exists();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_path)
            .expect("could not open or create metadata.csv");
        let mut writer = Writer::from_writer(file);
        if is_new || metadata(&csv_path).map(|m| m.len() == 0).unwrap_or(true) {
            writer.write_record(&["episode","mean_score", "mean_duration", "total_elapsed_secs","interval_elapsed_secs",
                "num_episodes","episode_stop","gamma","alpha",
                "start_epsilon","final_epsilon","weight_decay"]).unwrap();
            writer.flush().unwrap();
        }

        let now = Instant::now();
        BaseLogger { writer, start_time: now, last_log_time: now, run_dir, input_dir, env_name }
    }

    /// Log base metrics. Returns the run directory for further use.
    pub fn make_base(&mut self, episode: usize, mean_score: f32, mean_duration: std::time::Duration
    ) -> RecordBase {
        use std::time::Instant;
        let now      = Instant::now();
        let total    = now.duration_since(self.start_time).as_secs_f64();
        let interval = now.duration_since(self.last_log_time).as_secs_f64();
        self.last_log_time = now;
        RecordBase { episode, mean_score, mean_duration, total_elapsed_secs: total, interval_elapsed_secs: interval }
    }


    /// Expose run directory path
    pub fn run_dir(&self) -> &PathBuf { &self.run_dir }
    
    pub fn run_input_dir(&self) -> &PathBuf { &self.input_dir }
}