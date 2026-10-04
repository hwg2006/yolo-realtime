//! 后处理：解析 YOLO 输出张量、阈值过滤、letterbox 坐标还原与类别内 NMS。

use crate::types::{Detection, NUM_CLASSES};

/// 解析输出张量 `[1, 4+nc, anchors]` 为检测框列表。
///
/// 输出布局为通道优先：前 4 通道是 cx/cy/w/h，其后为各类别分数（无 objectness）。
/// 坐标经 letterbox 反变换映射回原图，并按置信度做类别内 NMS。
#[allow(clippy::too_many_arguments)]
pub fn postprocess(
    data: &[f32],
    anchors: usize,
    scale: f32,
    dw: f32,
    dh: f32,
    orig: (u32, u32),
    conf_thres: f32,
    iou_thres: f32,
) -> Vec<Detection> {
    let (ow, oh) = (orig.0 as f32, orig.1 as f32);
    let mut raw: Vec<Detection> = Vec::new();

    for a in 0..anchors {
        // 取最大类别分数（YOLOv8/11 无 objectness）
        let mut best = 0usize;
        let mut best_score = 0.0f32;
        for c in 0..NUM_CLASSES {
            let s = data[(4 + c) * anchors + a];
            if s > best_score {
                best_score = s;
                best = c;
            }
        }
        if best_score < conf_thres {
            continue;
        }
        let cx = data[a];
        let cy = data[anchors + a];
        let bw = data[2 * anchors + a];
        let bh = data[3 * anchors + a];

        // letterbox -> 原图坐标
        let x1 = ((cx - bw / 2.0) - dw) / scale;
        let y1 = ((cy - bh / 2.0) - dh) / scale;
        let x2 = ((cx + bw / 2.0) - dw) / scale;
        let y2 = ((cy + bh / 2.0) - dh) / scale;

        raw.push(Detection {
            x1: x1.clamp(0.0, ow),
            y1: y1.clamp(0.0, oh),
            x2: x2.clamp(0.0, ow),
            y2: y2.clamp(0.0, oh),
            cls: best,
            conf: best_score,
        });
    }

    nms(raw, iou_thres)
}

/// 类别内 NMS：同类别互相抑制，不同类别互不影响。
pub fn nms(mut dets: Vec<Detection>, iou_thres: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<Detection> = Vec::new();
    let mut removed = vec![false; dets.len()];
    for i in 0..dets.len() {
        if removed[i] {
            continue;
        }
        keep.push(dets[i].clone());
        for j in (i + 1)..dets.len() {
            if removed[j] || dets[j].cls != dets[i].cls {
                continue;
            }
            if iou(&dets[i], &dets[j]) > iou_thres {
                removed[j] = true;
            }
        }
    }
    keep
}

/// 交并比。
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, cls: usize, conf: f32) -> Detection {
        Detection { x1, y1, x2, y2, cls, conf }
    }

    #[test]
    fn iou_identical_is_one() {
        let a = det(0.0, 0.0, 10.0, 10.0, 0, 0.9);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_is_zero() {
        let a = det(0.0, 0.0, 10.0, 10.0, 0, 0.9);
        let b = det(20.0, 20.0, 30.0, 30.0, 0, 0.9);
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn iou_half_overlap() {
        // 两个 10x10 框水平重叠一半：交 50，并 150 -> 1/3
        let a = det(0.0, 0.0, 10.0, 10.0, 0, 0.9);
        let b = det(5.0, 0.0, 15.0, 10.0, 0, 0.9);
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_highest_conf() {
        let a = det(0.0, 0.0, 10.0, 10.0, 0, 0.9);
        let b = det(1.0, 1.0, 11.0, 11.0, 0, 0.5); // 与 a 高度重叠
        let kept = nms(vec![b, a], 0.45);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].conf - 0.9).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_different_classes() {
        let a = det(0.0, 0.0, 10.0, 10.0, 0, 0.9);
        let b = det(1.0, 1.0, 11.0, 11.0, 1, 0.5);
        assert_eq!(nms(vec![a, b], 0.45).len(), 2);
    }

    #[test]
    fn postprocess_maps_coords_back() {
        // anchors=1, 84 通道：cx=cy=208, w=h=416, 类别 0 分数 0.9
        let anchors = 1usize;
        let mut data = vec![0.0f32; (4 + NUM_CLASSES) * anchors];
        data[0] = 208.0; // cx
        data[1] = 208.0; // cy
        data[2] = 416.0; // w
        data[3] = 416.0; // h
        data[4] = 0.9; // class 0 score
        let dets = postprocess(&data, anchors, 1.0, 0.0, 0.0, (416, 416), 0.25, 0.45);
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert_eq!(d.cls, 0);
        assert!((d.x1 - 0.0).abs() < 1e-3);
        assert!((d.y1 - 0.0).abs() < 1e-3);
        assert!((d.x2 - 416.0).abs() < 1e-3);
        assert!((d.y2 - 416.0).abs() < 1e-3);
    }

    #[test]
    fn postprocess_filters_low_conf() {
        let anchors = 1usize;
        let mut data = vec![0.0f32; (4 + NUM_CLASSES) * anchors];
        data[4] = 0.1; // 低于阈值
        assert!(postprocess(&data, anchors, 1.0, 0.0, 0.0, (416, 416), 0.25, 0.45).is_empty());
    }
}
