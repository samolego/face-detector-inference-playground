use image::{Rgb, RgbImage};

pub fn draw_rectangle(image: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32) {
    let color = Rgb([255, 0, 0]);
    let thickness = 3;

    // Draw horizontal lines
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

    // Draw vertical lines
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
