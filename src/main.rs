use image::{imageops::FilterType, Rgb, RgbImage};
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
    let width = 640;
    let height = 640;
    let mut input_data = Vec::with_capacity(width * height * 3);

    // Match Python's preprocessing:
    // blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size,
    //     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
    // where input_std = 128.0 and input_mean = 127.5
    for c in 0..3 {
        for i in 0..height {
            for j in 0..width {
                let index = (i * width + j) * 3;
                let pixel_value = raw[index + c] as f32;
                // Normalize using the same values as Python
                input_data.push((pixel_value - 127.5) / 128.0);
            }
        }
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

    let confidence_threshold = 0.5;

    // Revised decode_box without using +1 for widths/heights.
    fn decode_box(points: [f32; 2], distance: [f32; 4]) -> [f32; 4] {
        // Python equivalent of distance2bbox:
        // x1 = points[:, 0] - distance[:, 0]
        // y1 = points[:, 1] - distance[:, 1]
        // x2 = points[:, 0] + distance[:, 2]
        // y2 = points[:, 1] + distance[:, 3]

        let x1 = points[0] - distance[0];
        let y1 = points[1] - distance[1];
        let x2 = points[0] + distance[2];
        let y2 = points[1] + distance[3];

        [x1, y1, x2, y2]
    }

    // Process one scale branch.
    let mut process_scale = |scores: &[f32],
                             bboxes: &[f32],
                             stride: u32|
     -> Result<(), Box<dyn std::error::Error>> {
        let grid_size = 640 / stride; // assuming no remainder
        let cells = (grid_size * grid_size) as usize;
        let num_anchors = scores.len() / cells;
        let scales = match stride {
            8 => vec![2.0, 1.0],
            16 => vec![8.0, 4.0],
            32 => vec![32.0, 16.0],
            _ => vec![1.0],
        };
        if num_anchors != scales.len() {
            return Err(format!(
                "For stride {} we expected {} anchors but got {}",
                stride,
                scales.len(),
                num_anchors
            )
            .into());
        }
        let base_size: f32 = 16.0;
        for cell in 0..cells {
            let row = cell as u32 / grid_size;
            let col = cell as u32 % grid_size;

            for a in 0..num_anchors {
                let index = cell * num_anchors + a;
                let confidence = scores[index];
                if confidence > confidence_threshold {
                    // Generate grid points (matching Python's anchor_centers generation)
                    let cx = (col as f32) * (stride as f32);
                    let cy = (row as f32) * (stride as f32);

                    // Get bbox predictions
                    let start = index * 4;
                    // Scale the entire bbox predictions by stride
                    let scaled_deltas = [
                        bboxes[start + 0] * (stride as f32),
                        bboxes[start + 1] * (stride as f32),
                        bboxes[start + 2] * (stride as f32),
                        bboxes[start + 3] * (stride as f32),
                    ];

                    // Also scale the center points by stride to match Python's behavior
                    let scaled_points = [cx, cy];
                    let decoded = decode_box(scaled_points, scaled_deltas);

                    println!(
                        "Detection (stride {}): Conf: {:.4}, decoded_box: [{:.1}, {:.1}, {:.1}, {:.1}]",
                        stride, confidence, decoded[0], decoded[1], decoded[2], decoded[3]
                    );

                    let img_width = output_image.width() as f32;
                    let img_height = output_image.height() as f32;
                    let x1 = decoded[0].max(0.0).min(img_width - 1.0) as u32;
                    let y1 = decoded[1].max(0.0).min(img_height - 1.0) as u32;
                    let x2 = decoded[2].max(0.0).min(img_width - 1.0) as u32;
                    let y2 = decoded[3].max(0.0).min(img_height - 1.0) as u32;

                    println!(
                        "Final rectangle coordinates (after bounds check): x1={}, y1={}, x2={}, y2={}",
                        x1, y1, x2, y2
                    );
                    draw_rectangle(&mut output_image, x1, y1, x2, y2);
                }
            }
        }
        Ok(())
    };

    let scores08_slice = scores08.as_slice().ok_or("Failed to get scores08 slice")?;
    let bboxes08_slice = bboxes08.as_slice().ok_or("Failed to get bboxes08 slice")?;
    process_scale(scores08_slice, bboxes08_slice, 8)?;

    let scores16_slice = scores16.as_slice().ok_or("Failed to get scores16 slice")?;
    let bboxes16_slice = bboxes16.as_slice().ok_or("Failed to get bboxes16 slice")?;
    process_scale(scores16_slice, bboxes16_slice, 16)?;

    let scores32_slice = scores32.as_slice().ok_or("Failed to get scores32 slice")?;
    let bboxes32_slice = bboxes32.as_slice().ok_or("Failed to get bboxes32 slice")?;
    process_scale(scores32_slice, bboxes32_slice, 32)?;

    output_image.save("output.jpg")?;

    Ok(())
}

fn draw_rectangle(image: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32) {
    let color = Rgb([255, 0, 0]);
    let thickness = 3;

    // Draw horizontal lines with thickness.
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

    // Draw vertical lines with thickness.
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
