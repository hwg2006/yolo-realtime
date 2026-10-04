//! YOLO11n 推理库（Rust + ONNX Runtime）。
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
//! 模块划分：`types`(数据) / `paths`(路径与环境) / `preprocess`(预处理) /
//! `postprocess`(后处理) / `draw`(可视化) / `detector`(会话与推理)。
//! 二进制入口 `main.rs` 只做 CLI 解析与模块组装。

pub mod detector;
pub mod draw;
pub mod paths;
pub mod postprocess;
pub mod preprocess;
pub mod types;

pub use detector::Detector;
pub use draw::annotate;
pub use paths::{collect_images, default_model, load_font, project_root, setup_ort_dylib};
pub use postprocess::{iou, nms, postprocess};
pub use preprocess::letterbox;
pub use types::{Detection, Timings, COCO_NAMES, NUM_CLASSES};
