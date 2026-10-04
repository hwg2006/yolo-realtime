//! CLI 入口：解析命令行参数，组装检测器并分发到各子命令。
//!
//! 用法:
//!   yolo-infer image 图片.jpg --out out/
//!   yolo-infer dir   图片目录 --out out/ --limit 50
//!   yolo-infer bench 图片目录 --rounds 3
//!   yolo-infer camera 0 --frames 300        # 需要 --features camera 编译
//!
//! 具体推理逻辑见库模块（`preprocess` / `postprocess` / `draw` / `detector`）。

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use image::RgbImage;

use yolo_infer::{
    annotate, collect_images, default_model, load_font, project_root, setup_ort_dylib, Detector,
    Timings,
};

#[derive(Parser)]
#[command(name = "yolo-infer", about = "YOLO11n 推理（Rust + ONNX Runtime）")]
struct Cli {
    /// ONNX 模型路径
    #[arg(long, default_value_os_t = default_model())]
    model: PathBuf,
    /// 推理输入尺寸（需与导出的 ONNX 一致）
    #[arg(long, default_value_t = 416)]
    size: usize,
    /// 置信度阈值
    #[arg(long, default_value_t = 0.25)]
    conf: f32,
    /// NMS IoU 阈值
    #[arg(long, default_value_t = 0.45)]
    iou: f32,
    /// intra-op 线程数
    #[arg(long, default_value_t = 8)]
    threads: usize,
    /// 结果输出目录
    #[arg(long)]
    out: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 单张图片
    Image { path: PathBuf },
    /// 目录下多张图片
    Dir {
        path: PathBuf,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// 压测：重复推理并统计各阶段耗时
    Bench {
        path: PathBuf,
        #[arg(long, default_value_t = 3)]
        rounds: usize,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// 摄像头实时推理
    #[cfg(feature = "camera")]
    Camera {
        #[arg(default_value_t = 0)]
        index: u32,
        #[arg(long, default_value_t = 300)]
        frames: u32,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    setup_ort_dylib();
    let cli = Cli::parse();
    let mut det = Detector::new(&cli.model, cli.size, cli.threads, cli.conf, cli.iou)?;
    let font = load_font();
    println!(
        "[yolo-infer] model={} size={} conf={} iou={} threads={}",
        cli.model.display(),
        cli.size,
        cli.conf,
        cli.iou,
        cli.threads
    );

    match cli.cmd {
        Cmd::Image { path } => {
            let img = image::open(&path)?.to_rgb8();
            let (dets, t) = det.infer(&img)?;
            let vis = annotate(&img, &dets, font.as_ref());
            let out = cli.out.clone().unwrap_or_else(|| project_root().join("runs/detect"));
            std::fs::create_dir_all(&out)?;
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
        }
        Cmd::Dir { path, limit } => {
            let files = collect_images(&path, limit)?;
            let out = cli.out.clone().unwrap_or_else(|| project_root().join("runs/detect/rust_dir"));
            std::fs::create_dir_all(&out)?;
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
                if cli.out.is_some() {
                    let vis = annotate(&img, &dets, font.as_ref());
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
        }
        Cmd::Bench { path, rounds, limit } => {
            let files = collect_images(&path, limit)?;
            let imgs: Vec<RgbImage> = files
                .iter()
                .map(|f| image::open(f).map(|i| i.to_rgb8()))
                .collect::<Result<_, _>>()?;
            if imgs.is_empty() {
                anyhow::bail!("目录下没有图片: {}", path.display());
            }
            // 预热
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
        }
        #[cfg(feature = "camera")]
        Cmd::Camera { index, frames, out } => run_camera(&mut det, &font, index, frames, out, &cli)?,
    }
    Ok(())
}

#[cfg(feature = "camera")]
fn run_camera(
    det: &mut Detector,
    font: &Option<Vec<u8>>,
    index: u32,
    frames: u32,
    out: Option<PathBuf>,
    cli: &Cli,
) -> Result<()> {
    use nokhwa::pixel_format::RgbFormat;
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::Camera;

    let fmt = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut cam = Camera::new(CameraIndex::Index(index), fmt)
        .with_context(|| format!("打开摄像头 {index} 失败"))?;
    cam.open_stream()?;
    if let Some(o) = &out {
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
        if let Some(o) = &out {
            if i == 0 {
                annotate(&img, &dets, font.as_ref()).save(o.join("camera_first.jpg"))?;
            }
        }
        let _ = cli;
    }
    println!("[camera] 完成，平均 FPS≈{fps:.1}");
    Ok(())
}
