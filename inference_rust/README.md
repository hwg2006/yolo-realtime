# inference_rust —— Rust 原生推理

基于 `ort`（ONNX Runtime）+ `image` 的原生推理，全流程与 Python 版对齐，用于性能优化对比。

代码按职责拆分为库（`src/lib.rs`）+ 二进制入口（`src/main.rs`），入口只做 CLI 解析与模块组装。

| 文件 | 作用 |
| --- | --- |
| [src/main.rs](./src/main.rs) | CLI 入口：参数解析与子命令分发（`image` / `dir` / `bench` / `camera`） |
| [src/detector.rs](./src/detector.rs) | `Detector`：加载 ONNX 会话并执行单帧推理 |
| [src/preprocess.rs](./src/preprocess.rs) | letterbox 缩放 + 归一化 + CHW 排布 |
| [src/postprocess.rs](./src/postprocess.rs) | 输出解析、阈值过滤、坐标还原、类别内 NMS |
| [src/draw.rs](./src/draw.rs) | 类别配色与检测框 / 标签绘制 |
| [src/types.rs](./src/types.rs) | 检测框、耗时统计与 COCO 类别表 |
| [src/paths.rs](./src/paths.rs) | 项目路径、onnxruntime 动态库、字体与图片枚举 |
| [Cargo.toml](./Cargo.toml) | 依赖与 release 优化配置（LTO、`opt-level=3`、`codegen-units=1`） |

## 运行

```powershell
cd inference_rust
cargo test                              # 运行后处理 / 预处理 / 绘制的单元测试
cargo build --release
$env:ORT_DYLIB_PATH = "..\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll"

.\target\release\yolo-infer.exe bench ..\datasets\coco128\images\train2017 --rounds 5 --limit 40
```

- 需已有 `models/yolo11n_coco128.onnx`
- `--out` 是全局参数，须写在子命令之前；`camera` 子命令需启用 `camera` feature
