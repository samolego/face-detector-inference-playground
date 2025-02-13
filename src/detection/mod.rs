pub mod bbox;
pub mod model;
pub mod preprocessing;

use crate::drawing::rectangle::draw_rectangle;
use image::RgbImage;
use ort::SessionOutputs;

#[derive(Debug)]
pub struct Detection {
    pub confidence: f32,
    pub bbox: [f32; 4],
}

pub fn process_detections(
    results: &SessionOutputs,
    confidence_threshold: f32,
) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
    let scores08 = results[0].try_extract_tensor::<f32>()?;
    let scores16 = results[1].try_extract_tensor::<f32>()?;
    let scores32 = results[2].try_extract_tensor::<f32>()?;
    let bboxes08 = results[3].try_extract_tensor::<f32>()?;
    let bboxes16 = results[4].try_extract_tensor::<f32>()?;
    let bboxes32 = results[5].try_extract_tensor::<f32>()?;

    let mut detections = Vec::new();

    // Process each scale
    process_scale_detections(
        scores08.as_slice().ok_or("Failed to get scores08 slice")?,
        bboxes08.as_slice().ok_or("Failed to get bboxes08 slice")?,
        8,
        confidence_threshold,
        &mut detections,
    )?;

    process_scale_detections(
        scores16.as_slice().ok_or("Failed to get scores16 slice")?,
        bboxes16.as_slice().ok_or("Failed to get bboxes16 slice")?,
        16,
        confidence_threshold,
        &mut detections,
    )?;

    process_scale_detections(
        scores32.as_slice().ok_or("Failed to get scores32 slice")?,
        bboxes32.as_slice().ok_or("Failed to get bboxes32 slice")?,
        32,
        confidence_threshold,
        &mut detections,
    )?;

    Ok(detections)
}

fn process_scale_detections(
    scores: &[f32],
    bboxes: &[f32],
    stride: u32,
    confidence_threshold: f32,
    detections: &mut Vec<Detection>,
) -> Result<(), Box<dyn std::error::Error>> {
    let grid_size = 640 / stride;
    let cells = (grid_size * grid_size) as usize;
    let scales = bbox::get_scales(stride);
    let num_anchors = scores.len() / cells;

    if num_anchors != scales.len() {
        return Err(format!(
            "For stride {} we expected {} anchors but got {}",
            stride,
            scales.len(),
            num_anchors
        )
        .into());
    }

    for cell in 0..cells {
        let row = cell as u32 / grid_size;
        let col = cell as u32 % grid_size;

        for a in 0..num_anchors {
            let index = cell * num_anchors + a;
            let confidence = scores[index];

            if confidence > confidence_threshold {
                let cx = (col as f32) * (stride as f32);
                let cy = (row as f32) * (stride as f32);

                let start = index * 4;
                let scaled_deltas = [
                    bboxes[start + 0] * (stride as f32),
                    bboxes[start + 1] * (stride as f32),
                    bboxes[start + 2] * (stride as f32),
                    bboxes[start + 3] * (stride as f32),
                ];

                let scaled_points = [cx, cy];
                let bbox = bbox::decode_box(scaled_points, scaled_deltas);

                detections.push(Detection { confidence, bbox });
            }
        }
    }

    Ok(())
}

pub fn draw_detections(
    output_image: &mut RgbImage,
    detections: &[Detection],
) -> Result<(), Box<dyn std::error::Error>> {
    let img_width = output_image.width() as f32;
    let img_height = output_image.height() as f32;

    for detection in detections {
        let x1 = detection.bbox[0].max(0.0).min(img_width - 1.0) as u32;
        let y1 = detection.bbox[1].max(0.0).min(img_height - 1.0) as u32;
        let x2 = detection.bbox[2].max(0.0).min(img_width - 1.0) as u32;
        let y2 = detection.bbox[3].max(0.0).min(img_height - 1.0) as u32;

        draw_rectangle(output_image, x1, y1, x2, y2);
    }

    Ok(())
}
