//! CLI 子命令实现：把库能力（检测 / 绘制 / 路径）编排成 image、dir、bench、camera 四个用例。
//!
//! `main.rs` 只负责解析参数、构建 `Detector` 并分发到这里，业务编排集中在本模块。

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use image::RgbImage;

#[cfg(feature = "camera")]
use anyhow::Context;

use crate::detector::Detector;
use crate::draw::annotate;
use crate::paths::{collect_images, project_root};
use crate::types::Timings;

/// 解析输出目录：未显式指定时落到仓库内的默认相对路径，并确保目录存在。
fn resolve_out_dir(out: Option<&Path>, default_rel: &str) -> Result<PathBuf> {
    let dir = out
        .map(Path::to_path_buf)
        .unwrap_or_else(|| project_root().join(default_rel));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// 单张图片：推理一次，把可视化结果写入输出目录。
pub fn run_image(
    det: &mut Detector,
    font: Option<&Vec<u8>>,
    path: &Path,
    out: Option<&Path>,
) -> Result<()> {
    let img = image::open(path)?.to_rgb8();
    let (dets, t) = det.infer(&img)?;
    let vis = annotate(&img, &dets, font);
    let out = resolve_out_dir(out, "runs/detect")?;
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let dst = out.join(format!("rust_{name}"));
    vis.save(&dst)?;
    println!(
        "[yolo-infer] 检测到 {} 个目标  pre={:.1}ms infer={:.1}ms post={:.1}ms -> {}",
        dets.len(),
        t.pre,
        t.infer,
        t.post,
        dst.display()
    );
    Ok(())
}

/// 目录下多张图片：逐张推理并统计平均耗时；仅在显式指定 `--out` 时落盘。
pub fn run_dir(
    det: &mut Detector,
    font: Option<&Vec<u8>>,
    path: &Path,
    limit: usize,
    out: Option<&Path>,
) -> Result<()> {
    let files = collect_images(path, limit)?;
    let save = out.is_some();
    let out = resolve_out_dir(out, "runs/detect/rust_dir")?;
    let mut acc = Timings::default();
    let t0 = Instant::now();
    let mut n_det = 0usize;
    for f in &files {
        let img = image::open(f)?.to_rgb8();
        let (dets, t) = det.infer(&img)?;
        n_det += dets.len();
        acc.pre += t.pre;
        acc.infer += t.infer;
        acc.post += t.post;
        if save {
            let vis = annotate(&img, &dets, font);
            let name = f.file_name().unwrap_or_default().to_string_lossy();
            vis.save(out.join(format!("rust_{name}")))?;
        }
    }
    let n = files.len().max(1) as f64;
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "[yolo-infer] {} 张图, 共 {} 个目标 | 平均 pre={:.1}ms infer={:.1}ms post={:.1}ms | 端到端 {:.1} FPS",
        files.len(),
        n_det,
        acc.pre / n,
        acc.infer / n,
        acc.post / n,
        files.len() as f64 / wall
    );
    Ok(())
}

/// 压测：重复推理并统计各阶段平均耗时与吞吐。
pub fn run_bench(det: &mut Detector, path: &Path, rounds: usize, limit: usize) -> Result<()> {
    let files = collect_images(path, limit)?;
    let imgs: Vec<RgbImage> = files
        .iter()
        .map(|f| image::open(f).map(|i| i.to_rgb8()))
        .collect::<Result<_, _>>()?;
    if imgs.is_empty() {
        anyhow::bail!("目录下没有图片: {}", path.display());
    }
    // 预热，避免首帧的惰性初始化计入统计
    let _ = det.infer(&imgs[0])?;
    let mut acc = Timings::default();
    let mut n = 0f64;
    let t0 = Instant::now();
    for _ in 0..rounds {
        for img in &imgs {
            let (_, t) = det.infer(img)?;
            acc.pre += t.pre;
            acc.infer += t.infer;
            acc.post += t.post;
            n += 1.0;
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let per = |x: f64| x / n;
    println!(
        "[bench] rounds={} images={} runs={} | 平均 pre={:.2}ms infer={:.2}ms post={:.2}ms 合计={:.2}ms | 纯推理 {:.1} FPS | 端到端(含解码) {:.1} FPS",
        rounds,
        imgs.len(),
        n as usize,
        per(acc.pre),
        per(acc.infer),
        per(acc.post),
        per(acc.pre + acc.infer + acc.post),
        1000.0 / per(acc.pre + acc.infer + acc.post),
        n / wall
    );
    Ok(())
}

/// 摄像头实时推理：连续抓帧 → 推理 → 可选落盘首帧（需 `--features camera`）。
#[cfg(feature = "camera")]
pub fn run_camera(
    det: &mut Detector,
    font: Option<&Vec<u8>>,
    index: u32,
    frames: u32,
    out: Option<&Path>,
) -> Result<()> {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::Camera;

    let fmt = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut cam = Camera::new(CameraIndex::Index(index), fmt)
        .with_context(|| format!("打开摄像头 {index} 失败"))?;
    cam.open_stream()?;
    if let Some(o) = out {
        std::fs::create_dir_all(o)?;
    }
    println!("[camera] 已打开摄像头 {index}, 开始实时推理 ({frames} 帧)");

    let mut fps = 0.0f64;
    let mut t_last = Instant::now();
    for i in 0..frames {
        let frame = cam.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;
        let img = RgbImage::from_raw(decoded.width(), decoded.height(), decoded.into_raw())
            .context("构造图像失败")?;

        let (dets, t) = det.infer(&img)?;
        let now = Instant::now();
        let inst = 1.0 / now.duration_since(t_last).as_secs_f64().max(1e-6);
        fps = if fps == 0.0 { inst } else { fps * 0.9 + inst * 0.1 };
        t_last = now;

        if i % 30 == 0 {
            println!(
                "[camera] frame={} fps={:.1} dets={} infer={:.1}ms",
                i,
                fps,
                dets.len(),
                t.infer
            );
        }
        if let Some(o) = out {
            if i == 0 {
                annotate(&img, &dets, font).save(o.join("camera_first.jpg"))?;
            }
        }
    }
    println!("[camera] 完成，平均 FPS≈{fps:.1}");
    Ok(())
}
