use std::rc::Rc;

use ndarray::ArrayBase;
use ort::{GraphOptimizationLevel, Session, SessionOutputs};

pub struct DetectionModel {
    session: Session,
}

impl DetectionModel {
    pub fn new(config: Rc<crate::config::Config>) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.intra_threads)?
            .commit_from_file(&config.model_path)?;

        Ok(Self { session })
    }

    pub fn predict(
        &self,
        input_array: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>,
    ) -> Result<SessionOutputs, Box<dyn std::error::Error>> {
        Ok(self
            .session
            .run(ort::inputs!["input.1" => input_array.view()]?)?)
    }
}
