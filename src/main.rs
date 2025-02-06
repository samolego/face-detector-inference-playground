use image::{imageops::FilterType, GenericImageView, Rgb, RgbImage};
use ndarray::Array;
use ort::{inputs, CPUExecutionProvider, GraphOptimizationLevel, Session};

pub mod models;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ort::init()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    println!("Loading model");

    let model_path = "res/models/buffalo_l_detection.onnx";
    let input_shape = [1, 3, 640, 640];

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    println!("Model loaded");

    let image_path = std::env::args().nth(1).expect("Image path is required");
    println!("Loading image");
    let img = image::open(image_path)?;
    let resized = img.resize_exact(640, 640, FilterType::CatmullRom);
    let mut output_image = resized.to_rgb8();

    let rgb_image = resized.to_rgb8();
    let raw = rgb_image.into_raw();
    let mut input_data = Vec::with_capacity(640 * 640 * 3);

    for pixel in raw.chunks(3) {
        input_data.push(pixel[2] as f32);
        input_data.push(pixel[1] as f32);
        input_data.push(pixel[0] as f32);
    }

    let input_array = Array::from_shape_vec(input_shape, input_data).unwrap();

    println!("Starting inference");
    let result = model.run(inputs!["input.1" => input_array.view()]?)?;
    println!("Inference done");

    let scores08 = result[0].try_extract_tensor::<f32>()?;
    let scores16 = result[1].try_extract_tensor::<f32>()?;
    let scores32 = result[2].try_extract_tensor::<f32>()?;
    let bboxes08 = result[3].try_extract_tensor::<f32>()?;
    let bboxes16 = result[4].try_extract_tensor::<f32>()?;
    let bboxes32 = result[5].try_extract_tensor::<f32>()?;

    let confidence_threshold = 0.84;

    let mut process_scale = |scores: &[f32],
                             bboxes: &[f32],
                             shape: &[usize]|
     -> Result<(), Box<dyn std::error::Error>> {
        let num_boxes = shape[0];
        for i in 0..num_boxes {
            let confidence = scores[i];
            if confidence > confidence_threshold {
                // The model outputs normalized coordinates, multiply by 640 to get pixel coordinates
                let center_x = bboxes[i * 4 + 0] * 640.0;
                let center_y = bboxes[i * 4 + 1] * 640.0;
                let width = bboxes[i * 4 + 2] * 640.0;
                let height = bboxes[i * 4 + 3] * 640.0;

                println!("Detection. Confidence: {confidence}, center_x: {center_x}, center_y: {center_y}, width: {width}, height: {height}");

                let x1 = (center_x - width / 2.0).max(0.0) as u32;
                let y1 = (center_y - height / 2.0).max(0.0) as u32;
                let x2 = (center_x + width / 2.0) as u32;
                let y2 = (center_y + height / 2.0) as u32;

                let img_width = output_image.width();
                let img_height = output_image.height();
                let x1 = x1.min(img_width - 1);
                let y1 = y1.min(img_height - 1);
                let x2 = x2.min(img_width - 1);
                let y2 = y2.min(img_height - 1);

                draw_rectangle(&mut output_image, x1, y1, x2, y2);
            }
        }
        Ok(())
    };

    let scores08_slice = scores08.as_slice().ok_or("Failed to get scores08 slice")?;
    let bboxes08_slice = bboxes08.as_slice().ok_or("Failed to get bboxes08 slice")?;
    process_scale(scores08_slice, bboxes08_slice, &[scores08.len(), 1])?;

    let scores16_slice = scores16.as_slice().ok_or("Failed to get scores16 slice")?;
    let bboxes16_slice = bboxes16.as_slice().ok_or("Failed to get bboxes16 slice")?;
    process_scale(scores16_slice, bboxes16_slice, &[scores16.len(), 1])?;

    let scores32_slice = scores32.as_slice().ok_or("Failed to get scores32 slice")?;
    let bboxes32_slice = bboxes32.as_slice().ok_or("Failed to get bboxes32 slice")?;
    process_scale(scores32_slice, bboxes32_slice, &[scores32.len(), 1])?;

    output_image.save("output.jpg")?;

    Ok(())
}

fn draw_rectangle(image: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32) {
    let color = Rgb([255, 0, 0]);
    let thickness = 3; // Number of pixels thick

    // Draw horizontal lines with thickness
    for dy in 0..thickness {
        let y1_thick = y1.saturating_add(dy);
        let y2_thick = y2.saturating_add(dy);

        for x in x1..=x2 {
            if x < image.width() {
                if y1_thick < image.height() {
                    image.put_pixel(x, y1_thick, color);
                }
                if y2_thick < image.height() {
                    image.put_pixel(x, y2_thick, color);
                }
            }
        }
    }

    // Draw vertical lines with thickness
    for dx in 0..thickness {
        let x1_thick = x1.saturating_add(dx);
        let x2_thick = x2.saturating_add(dx);

        for y in y1..=y2 {
            if y < image.height() {
                if x1_thick < image.width() {
                    image.put_pixel(x1_thick, y, color);
                }
                if x2_thick < image.width() {
                    image.put_pixel(x2_thick, y, color);
                }
            }
        }
    }
}
