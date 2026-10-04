//! 路径解析与运行环境准备：项目根目录、默认模型、onnxruntime 动态库、字体、图片枚举。

use std::path::{Path, PathBuf};

use anyhow::Result;

/// 工程根目录（`inference_rust/` 的上一级）。
pub fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("project root")
        .to_path_buf()
}

/// 默认 ONNX 模型路径。
pub fn default_model() -> PathBuf {
    project_root().join("models").join("yolo11n_coco128.onnx")
}

/// 让 ort 的 `load-dynamic` 找到 onnxruntime.dll（优先复用 venv 里已下载的那份）。
pub fn setup_ort_dylib() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    let candidates = [
        project_root().join(".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"),
        project_root().join("onnxruntime.dll"),
    ];
    for c in candidates {
        if c.exists() {
            std::env::set_var("ORT_DYLIB_PATH", &c);
            return;
        }
    }
}

/// 读取绘制标签用的字体（缺失时返回 None，绘制时自动跳过文字）。
pub fn load_font() -> Option<Vec<u8>> {
    let p = project_root().join(".ultralytics/Ultralytics/Arial.ttf");
    std::fs::read(p).ok()
}

/// 收集目录下的图片文件（按名称排序，`limit>0` 时截断）。
pub fn collect_images(dir: &Path, limit: usize) -> Result<Vec<PathBuf>> {
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
