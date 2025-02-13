pub struct Config {
    pub model_path: String,
    pub input_shape: [i64; 4],
    pub input_mean: f32,
    pub input_std: f32,
    pub confidence_threshold: f32,
    pub intra_threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: "det_10g.onnx".to_string(),
            input_shape: [1, 3, 640, 640],
            input_mean: 127.5,
            input_std: 128.0,
            confidence_threshold: 0.5,
            intra_threads: 4,
        }
    }
}
