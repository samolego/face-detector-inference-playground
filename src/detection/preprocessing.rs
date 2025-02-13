use crate::config::Config;
use image::RgbImage;
use ndarray::{Array, ArrayBase, OwnedRepr};

pub fn preprocess_image(
    rgb_image: &RgbImage,
    config: &Config,
) -> Result<ArrayBase<OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>, Box<dyn std::error::Error>> {
    let raw = rgb_image.as_raw();
    let width = rgb_image.width();
    let height = rgb_image.height();
    let mut input_data = Vec::with_capacity((width * height * 3) as usize);

    for c in 0..3 {
        for i in 0..height {
            for j in 0..width {
                let index = (i * width + j) * 3;
                let pixel_value = raw[index as usize + c] as f32;
                input_data.push((pixel_value - config.input_mean) / config.input_std);
            }
        }
    }

    let shape = [
        config.input_shape[0] as usize,
        config.input_shape[1] as usize,
        config.input_shape[2] as usize,
        config.input_shape[3] as usize,
    ];

    Ok(Array::from_shape_vec(shape, input_data)?)
}
