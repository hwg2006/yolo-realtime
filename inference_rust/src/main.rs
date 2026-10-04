//! YOLO11n 实时推理（Rust + ONNX Runtime）。
//!
//! 复用开源库完成重活：
//!   - `ort`        : ONNX Runtime 绑定，负责推理加速
//!   - `image`      : 图像解码 / 缩放
//!   - `imageproc`  : 画框 / 写字
//!   - `nokhwa`     : 摄像头采集（`--features camera` 时启用）
//!
//! 与 Python 版 `inference_python/realtime.py` 对应，但去掉了 PyTorch/Ultralytics
//! 运行时开销，只依赖 ONNX Runtime，单帧延迟更低。
//!
//! 用法:
//!   yolo-infer image 图片.jpg --out out/
//!   yolo-infer dir   图片目录 --out out/ --limit 50
//!   yolo-infer bench 图片目录 --rounds 3
//!   yolo-infer camera 0 --frames 300        # 需要 --features camera 编译

use std::path::{Path, PathBuf};
use std::time::Instant;

use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

const NUM_CLASSES: usize = 80;

const COCO_NAMES: [&str; NUM_CLASSES] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("project root")
        .to_path_buf()
}

/// 把 ort 的错误转成 anyhow::Error。
///
/// 直接 `?` 会失败：部分 ort 错误类型带资源泛型（如 `Error<SessionBuilder>`），
/// 而 `SessionBuilder` 含裸指针、不满足 `Send + Sync`，anyhow 无法自动转换。
/// `ort::Error<R>` 对所有 `R` 都实现了 `Display`，所以统一转成字符串即可。
#[inline]
fn oerr<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

fn default_model() -> PathBuf {
    project_root().join("models").join("yolo11n_coco128.onnx")
}

/// 让 ort 的 `load-dynamic` 找到 onnxruntime.dll（优先复用 venv 里已下载的那份）。
fn setup_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    let candidates = [
        project_root()
            .join(".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"),
        project_root().join("onnxruntime.dll"),
    ];
    for c in candidates {
        if c.exists() {
            std::env::set_var("ORT_DYLIB_PATH", &c);
            return;
        }
    }
}

#[derive(Clone, Debug)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    cls: usize,
    conf: f32,
}

#[derive(Default, Clone, Copy)]
struct Timings {
    pre: f64,
    infer: f64,
    post: f64,
}

struct Detector {
    session: Session,
    size: usize,
    conf_thres: f32,
    iou_thres: f32,
}

impl Detector {
    fn new(model: &Path, size: usize, threads: usize, conf: f32, iou: f32) -> Result<Self> {
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

    /// 返回 (检测框, 各阶段耗时 ms)
    fn infer(&mut self, img: &RgbImage) -> Result<(Vec<Detection>, Timings)> {
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

/// letterbox 到 size×size，返回 CHW float 数据 + 缩放比 + 左右/上下 padding
fn letterbox(img: &RgbImage, size: usize) -> (Vec<f32>, f32, f32, f32) {
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

fn postprocess(
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

/// 类别内 NMS
fn nms(mut dets: Vec<Detection>, iou_thres: f32) -> Vec<Detection> {
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

fn iou(a: &Detection, b: &Detection) -> f32 {
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

fn class_color(cls: usize) -> Rgb<u8> {
    // 依据类别 id 生成稳定的高对比颜色
    let h = (cls as f32 * 47.0) % 360.0;
    let (r, g, b) = hsl_to_rgb(h / 360.0, 0.85, 0.55);
    Rgb([r, g, b])
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
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

fn load_font() -> Option<Vec<u8>> {
    let p = project_root().join(".ultralytics/Ultralytics/Arial.ttf");
    std::fs::read(p).ok()
}

fn annotate(img: &RgbImage, dets: &[Detection], font_bytes: Option<&Vec<u8>>) -> RgbImage {
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

fn collect_images(dir: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase()),
                Some(ref e) if e == "jpg" || e == "jpeg" || e == "png" || e == "bmp"
            )
        })
        .collect();
    v.sort();
    if limit > 0 && v.len() > limit {
        v.truncate(limit);
    }
    Ok(v)
}

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
