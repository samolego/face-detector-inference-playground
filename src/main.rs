use image::{imageops::FilterType, Rgb, RgbImage};
use ndarray::{Array, Array2, ArrayView, ArrayView2};
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

    // Load image
    let image_path = std::env::args().nth(1).expect("Image path is required");

    // Load image
    println!("Loading image");
    let image = image::open(image_path)?;
    let resized_image = image.resize_exact(640, 640, FilterType::CatmullRom);

    // Retrieve dimensions from resized_image before cloning or moving it
    let original_width = resized_image.width() as f32;
    let original_height = resized_image.height() as f32;

    // Clone resized_image for inference before converting it
    let image_input = resized_image.clone().into_rgb32f();

    // Use the original resized_image for drawing
    let mut output_image = resized_image.to_rgb8();

    let mut normalized_input = Vec::with_capacity(input_shape.iter().product());
    for pixel in image_input.into_raw() {
        normalized_input.push((pixel - 127.5) / 128.0);
    }

    // Reshape to desired input shape
    let input_array = Array::from_shape_vec(input_shape, normalized_input).unwrap();

    println!("Starting inference");
    let result = model.run(inputs!["input.1" => input_array.view()]?)?;
    println!("Inference done");

    let scores08 = result[0].try_extract_tensor::<f32>()?;
    let scores16 = result[1].try_extract_tensor::<f32>()?;
    let scores32 = result[2].try_extract_tensor::<f32>()?;
    let bboxes08 = result[3].try_extract_tensor::<f32>()?;
    let bboxes16 = result[4].try_extract_tensor::<f32>()?;
    let bboxes32 = result[5].try_extract_tensor::<f32>()?;

    let confidence_threshold = 0.025;

    // Convert tensors to appropriate views
    let mut process_scale = |scores: &[f32],
                             bboxes: &[f32],
                             shape: &[usize]|
     -> Result<(), Box<dyn std::error::Error>> {
        let scores_array = Array2::from_shape_vec([shape[0], 1], scores.to_vec())?;
        let bboxes_array = Array2::from_shape_vec([shape[0], 4], bboxes.to_vec())?;

        process_detections(
            scores_array.view(),
            bboxes_array.view(),
            confidence_threshold,
            &mut output_image,
            original_width,
            original_height,
        )
    };

    // Process each scale
    let scores08_slice = scores08.as_slice().ok_or("Failed to get scores08 slice")?;
    let bboxes08_slice = bboxes08.as_slice().ok_or("Failed to get bboxes08 slice")?;
    process_scale(scores08_slice, bboxes08_slice, &[scores08.shape()[0], 1])?;

    let scores16_slice = scores16.as_slice().ok_or("Failed to get scores16 slice")?;
    let bboxes16_slice = bboxes16.as_slice().ok_or("Failed to get bboxes16 slice")?;
    process_scale(scores16_slice, bboxes16_slice, &[scores16.shape()[0], 1])?;

    let scores32_slice = scores32.as_slice().ok_or("Failed to get scores32 slice")?;
    let bboxes32_slice = bboxes32.as_slice().ok_or("Failed to get bboxes32 slice")?;
    process_scale(scores32_slice, bboxes32_slice, &[scores32.shape()[0], 1])?;

    output_image.save("output.jpg")?;

    Ok(())
}

fn process_detections(
    scores: ArrayView2<f32>,
    bboxes: ArrayView2<f32>,
    confidence_threshold: f32,
    output_image: &mut RgbImage,
    original_width: f32,
    original_height: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    for i in 0..bboxes.shape()[0] {
        let confidence = scores[[i, 0]];

        if confidence > confidence_threshold {
            // The model outputs center_x, center_y, width, height
            let center_x = bboxes[[i, 0]];
            let center_y = bboxes[[i, 1]];
            let width = bboxes[[i, 2]];
            let height = bboxes[[i, 3]];

            println!("Detection. Confidence: {confidence}, center_x: {center_x}, center_y: {center_y}, width: {width}, height: {height}");

            // Convert to corner coordinates
            let x1 = (center_x - width / 2.0) * original_width;
            let y1 = (center_y - height / 2.0) * original_height;
            let x2 = (center_x + width / 2.0) * original_width;
            let y2 = (center_y + height / 2.0) * original_height;

            // Ensure coordinates are within bounds
            let x1 = x1.max(0.0).min(original_width) as u32;
            let y1 = y1.max(0.0).min(original_height) as u32;
            let x2 = x2.max(0.0).min(original_width) as u32;
            let y2 = y2.max(0.0).min(original_height) as u32;

            draw_rectangle(output_image, x1, y1, x2, y2);
        }
    }
    Ok(())
}

fn draw_rectangle(image: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32) {
    let color = Rgb([255, 0, 0]);

    // Draw horizontal lines
    for x in x1..=x2 {
        if x < image.width() && y1 < image.height() {
            image.put_pixel(x, y1, color);
        }
        if x < image.width() && y2 < image.height() {
            image.put_pixel(x, y2, color);
        }
    }

    // Draw vertical lines
    for y in y1..=y2 {
        if x1 < image.width() && y < image.height() {
            image.put_pixel(x1, y, color);
        }
        if x2 < image.width() && y < image.height() {
            image.put_pixel(x2, y, color);
        }
    }
}
