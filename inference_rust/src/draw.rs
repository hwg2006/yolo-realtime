//! 可视化：按类别生成颜色并在图上绘制检测框与标签。

use ab_glyph::{FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;

use crate::types::{Detection, COCO_NAMES};

/// 依据类别 id 生成稳定的高对比颜色。
pub fn class_color(cls: usize) -> Rgb<u8> {
    let h = (cls as f32 * 47.0) % 360.0;
    let (r, g, b) = hsl_to_rgb(h / 360.0, 0.85, 0.55);
    Rgb([r, g, b])
}

/// HSL -> RGB（h/s/l 均为 0..1）。
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - (((h * 6.0) % 2.0) - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match (h * 6.0) as i32 % 6 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// 在原图副本上绘制所有检测框与标签，返回新图。
pub fn annotate(img: &RgbImage, dets: &[Detection], font_bytes: Option<&Vec<u8>>) -> RgbImage {
    let mut out = img.clone();
    let font = font_bytes.and_then(|b| FontRef::try_from_slice(b).ok());
    let scale = PxScale::from(16.0);

    for d in dets {
        let color = class_color(d.cls);
        let x1 = d.x1.max(0.0) as i32;
        let y1 = d.y1.max(0.0) as i32;
        let x2 = d.x2.max(0.0) as i32;
        let y2 = d.y2.max(0.0) as i32;
        let w = (x2 - x1).max(1) as u32;
        let h = (y2 - y1).max(1) as u32;
        draw_hollow_rect_mut(&mut out, Rect::at(x1, y1).of_size(w, h), color);

        if let Some(f) = &font {
            let label = format!("{} {:.2}", COCO_NAMES[d.cls], d.conf);
            let ty = (y1 - 18).max(0);
            draw_text_mut(&mut out, color, x1 + 1, ty, scale, f, &label);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsl_primary_colors() {
        assert_eq!(hsl_to_rgb(0.0, 1.0, 0.5), (255, 0, 0));
        assert_eq!(hsl_to_rgb(1.0 / 3.0, 1.0, 0.5), (0, 255, 0));
        assert_eq!(hsl_to_rgb(2.0 / 3.0, 1.0, 0.5), (0, 0, 255));
    }

    #[test]
    fn class_color_is_stable() {
        assert_eq!(class_color(3), class_color(3));
        assert_ne!(class_color(0), class_color(1));
    }

    #[test]
    fn annotate_draws_on_copy() {
        let img = RgbImage::from_pixel(64, 64, Rgb([0, 0, 0]));
        let d = Detection { x1: 10.0, y1: 10.0, x2: 40.0, y2: 40.0, cls: 0, conf: 0.9 };
        let out = annotate(&img, &[d], None);
        assert_eq!(out.dimensions(), img.dimensions());
        // 原图保持不变，副本上出现非黑像素
        assert_eq!(img.get_pixel(10, 10)[0], 0);
        assert!(out.pixels().any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0));
    }
}
