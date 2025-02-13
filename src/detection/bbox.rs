pub fn decode_box(points: [f32; 2], distance: [f32; 4]) -> [f32; 4] {
    let x1 = points[0] - distance[0];
    let y1 = points[1] - distance[1];
    let x2 = points[0] + distance[2];
    let y2 = points[1] + distance[3];

    [x1, y1, x2, y2]
}

pub fn get_scales(stride: u32) -> Vec<f32> {
    match stride {
        8 => vec![2.0, 1.0],
        16 => vec![8.0, 4.0],
        32 => vec![32.0, 16.0],
        _ => vec![1.0],
    }
}
