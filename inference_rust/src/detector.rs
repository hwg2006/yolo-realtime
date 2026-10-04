//! ONNX Runtime 会话封装：模型加载与单帧推理。

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use image::RgbImage;
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::postprocess::postprocess;
use crate::preprocess::letterbox;
use crate::types::{Detection, Timings};

/// 把 ort 的错误转成 anyhow::Error。
///
/// 直接 `?` 会失败：部分 ort 错误类型带资源泛型（如 `Error<SessionBuilder>`），
/// 而 `SessionBuilder` 含裸指针、不满足 `Send + Sync`，anyhow 无法自动转换。
/// `ort::Error<R>` 对所有 `R` 都实现了 `Display`，所以统一转成字符串即可。
#[inline]
fn oerr<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// YOLO 检测器，持有 ONNX Runtime 会话与推理参数。
pub struct Detector {
    session: Session,
    size: usize,
    conf_thres: f32,
    iou_thres: f32,
}

impl Detector {
    /// 加载模型并配置优化级别与线程数。
    pub fn new(model: &Path, size: usize, threads: usize, conf: f32, iou: f32) -> Result<Self> {
        let session = Session::builder()
            .map_err(oerr)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(oerr)?
            .with_intra_threads(threads.max(1))
            .map_err(oerr)?
            .commit_from_file(model)
            .map_err(oerr)
            .with_context(|| format!("加载 ONNX 失败: {}", model.display()))?;
        Ok(Self {
            session,
            size,
            conf_thres: conf,
            iou_thres: iou,
        })
    }

    /// 对单张 RGB 图推理，返回 (检测框, 各阶段耗时 ms)。
    pub fn infer(&mut self, img: &RgbImage) -> Result<(Vec<Detection>, Timings)> {
        let t0 = Instant::now();
        let (chw, scale, dw, dh) = letterbox(img, self.size);
        let t1 = Instant::now();

        let arr = Array4::from_shape_vec((1, 3, self.size, self.size), chw)?;
        let input = Tensor::from_array(arr).map_err(oerr)?;
        let outputs = self.session.run(ort::inputs!["images" => input]).map_err(oerr)?;
        let t2 = Instant::now();

        let out = outputs["output0"].try_extract_array::<f32>().map_err(oerr)?;
        let data = out.as_slice().expect("contiguous output");
        let anchors = out.shape()[2];
        let dets = postprocess(
            data,
            anchors,
            scale,
            dw,
            dh,
            img.dimensions(),
            self.conf_thres,
            self.iou_thres,
        );
        let t3 = Instant::now();

        Ok((
            dets,
            Timings {
                pre: t1.duration_since(t0).as_secs_f64() * 1e3,
                infer: t2.duration_since(t1).as_secs_f64() * 1e3,
                post: t3.duration_since(t2).as_secs_f64() * 1e3,
            },
        ))
    }
}
