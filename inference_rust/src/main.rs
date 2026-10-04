//! CLI 入口：解析命令行参数，组装检测器并分发到各子命令。
//!
//! 用法:
//!   yolo-infer image 图片.jpg --out out/
//!   yolo-infer dir   图片目录 --out out/ --limit 50
//!   yolo-infer bench 图片目录 --rounds 3
//!   yolo-infer camera 0 --frames 300        # 需要 --features camera 编译
//!
//! 各子命令的实现见库模块 `commands`，推理逻辑见 `preprocess` / `postprocess` /
//! `draw` / `detector`；本文件只做装配与分发。

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use yolo_infer::{commands, default_model, load_font, setup_ort_dylib, Detector};

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

    let out = cli.out.clone();
    match cli.cmd {
        Cmd::Image { path } => commands::run_image(&mut det, font.as_ref(), &path, out.as_deref())?,
        Cmd::Dir { path, limit } => {
            commands::run_dir(&mut det, font.as_ref(), &path, limit, out.as_deref())?
        }
        Cmd::Bench { path, rounds, limit } => commands::run_bench(&mut det, &path, rounds, limit)?,
        #[cfg(feature = "camera")]
        Cmd::Camera { index, frames, out } => {
            commands::run_camera(&mut det, font.as_ref(), index, frames, out.as_deref())?
        }
    }
    Ok(())
}
