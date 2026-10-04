//! 图像预处理：letterbox 缩放 + 归一化 + CHW 排布。

use image::RgbImage;

/// letterbox 到 `size×size`，返回 CHW float 数据 + 缩放比 + 左右/上下 padding。
///
/// 等比缩放并居中填充（灰色 114/255），保持长宽比避免形变；
/// padding 值用于后处理阶段把坐标映射回原图。
pub fn letterbox(img: &RgbImage, size: usize) -> (Vec<f32>, f32, f32, f32) {
    let (w, h) = img.dimensions();
    let r = (size as f32 / w as f32).min(size as f32 / h as f32);
    let nw = ((w as f32 * r).round() as u32).max(1);
    let nh = ((h as f32 * r).round() as u32).max(1);
    let dw = (size as u32 - nw) / 2;
    let dh = (size as u32 - nh) / 2;

    let resized = image::imageops::resize(img, nw, nh, image::imageops::FilterType::Triangle);
    let plane = size * size;
    let mut chw = vec![114.0f32 / 255.0; plane * 3];
    for y in 0..nh {
        for x in 0..nw {
            let p = resized.get_pixel(x, y);
            let xx = (x + dw) as usize;
            let yy = (y + dh) as usize;
            let idx = yy * size + xx;
            chw[idx] = p[0] as f32 / 255.0;
            chw[plane + idx] = p[1] as f32 / 255.0;
            chw[2 * plane + idx] = p[2] as f32 / 255.0;
        }
    }
    (chw, r, dw as f32, dh as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn letterbox_shape_and_range() {
        let img = RgbImage::from_pixel(640, 480, Rgb([255, 0, 0]));
        let (chw, r, dw, dh) = letterbox(&img, 416);
        assert_eq!(chw.len(), 3 * 416 * 416);
        assert!(chw.iter().all(|v| (0.0..=1.0).contains(v)));
        // 640 宽为长边，缩放比 = 416/640
        assert!((r - 416.0 / 640.0).abs() < 1e-6);
        // 宽占满，上下留白
        assert_eq!(dw, 0);
        assert!(dh > 0);
    }

    #[test]
    fn letterbox_pads_with_gray() {
        let img = RgbImage::from_pixel(640, 480, Rgb([255, 0, 0]));
        let (chw, _, _, _) = letterbox(&img, 416);
        // 左上角落在 padding 区，应为填充灰 114/255
        assert!((chw[0] - 114.0 / 255.0).abs() < 1e-6);
    }
}
