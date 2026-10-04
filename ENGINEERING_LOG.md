# 工程日志 · YOLO 实时目标检测全链路

> 本文记录从"接到任务"到"跑通全链路"的**思路演进、关键决策与踩坑复盘**。
> 不逐行讲代码，重点说明**为什么这么做**。

---

## 0. 任务与约束

**任务**（"二、yolo 的跑通"）：

1. 完成一次 Ultralytics 训练（参数自定）；
2. 实时推理部署：先用 Python 快速实现，再用 C++/Rust 优化性能——**参考开源库，不要自研一坨大的**；
3. 接入应用：Web（前端三件套 / Vue / React）+ 桌面内嵌（Electron / Tauri）。

**硬约束**：

- 机器**无 CUDA**，只有 Intel i7-13650HX + Intel 核显 → 一切以**纯 CPU 可跑**为前提。
- 时间/算力有限 → 训练必须能在几分钟内完成。
- "不重复造轮子" → 每一层都优先选成熟开源库。

---

## 1. 总体思路（贯穿全程的主线）

> **先跑通 → 再优化 → 最后接入应用。**

拆成一条可复现的链路：

```
训练(PyTorch/ultralytics) → 导出(ONNX) → Python 推理验证 → Rust 推理优化 → FastAPI 服务化 → Web/桌面接入
```

三个关键判断：

1. **用 ONNX 解耦训练与推理**：训练在 PyTorch，推理要跨语言。ONNX 是唯一被 Python / Rust / 浏览器运行时都广泛支持的中间格式，选它做"中间语言"。
2. **正确性优先于性能**：先写最短的 Python 代码把模型跑对（导出对不对、后处理对齐没有），确认无误后再上 Rust 做性能优化，避免一上来就陷入难调试的原生代码。
3. **推理只做一份**：把推理收敛到 FastAPI 服务，各端只做"采集 + 展示"。模型只加载一次，逻辑只维护一份。

---

## 2. 阶段推进

### 阶段一 · 训练与导出

**思路**：作业要的是"跑通"，不是"刷榜"。所以：

- 数据集选 **COCO128**——只有 128 张图，是 COCO 官方的冒烟测试集，CPU 上几分钟能跑完一次完整训练，又覆盖 80 类真实场景。
- 模型选 **YOLO11n**——nano 级，参数最少、CPU 最快。
- 参数围绕"CPU 可跑 + 结果可用"折中：`imgsz=416`（省算力，且与后续 ONNX 推理尺寸对齐）、`batch=16`、`epochs=30`、`cache=True`（小数据集全量缓存）。

**结果**：30 epochs 约 434 秒跑完，`mAP50 = 0.7111`、`mAP50-95 = 0.5529`，产出 `best.pt`。

**导出**：`opset=12 + imgsz=416 + simplify=True + dynamic=False`，得到 10.2 MB 的 `yolo11n_coco128.onnx`，输入 `[1,3,416,416]`、输出 `[1,84,3549]`；同时复制到 `models/` 统一管理，供后续所有模块引用。

> **决策点**：为什么固定 `dynamic=False`？
> 本项目只服务固定输入尺寸的实时场景，固定尺寸能触发更多图优化、推理路径最短；动态尺寸的灵活性此处用不上。

### 阶段二 · Python 快速推理（验证正确性）

**思路**：用最少的代码验证"模型导出是否正确、后处理是否对得上"。

- 直接用 `ultralytics` 的 `YOLO(weights).predict(source=..., stream=True)`，一份代码同时支持摄像头 / 视频 / 图片 / 目录。
- 提供 `--engine pt|onnx`，同一份脚本既能跑 `.pt` 也能跑 `.onnx`，方便对照。
- 滑动平均估算 FPS，`result.plot()` 直接出带框图。

**结果**：图片与视频均能正确检出目标（如 9 个），视频流平均 **≈ 41 FPS**。正确性确认，可以进入优化阶段。

### 阶段三 · Rust 原生推理（性能优化）

**思路**：优化**必须基于开源库**，把精力放在"业务级管线"而不是重写底层。选型：

| 关注点 | 选型 | 理由 |
| --- | --- | --- |
| 推理引擎 | `ort`（ONNX Runtime 官方 Rust 绑定） | 官方维护、性能与 C++ 版一致 |
| 图像 IO/缩放 | `image` | Rust 生态最成熟的图像库 |
| 画框/画字 | `imageproc` + `ab_glyph` | 与 `image` 无缝配合，不引入 OpenCV |
| 摄像头 | `nokhwa` | 跨平台，作为可选 feature |
| CLI | `clap`（derive） | 零手写解析 |

**实现要点**：手写 letterbox 预处理 → `ort` 推理 → 解码（YOLOv8/11 无 objectness，取 80 类最大分）→ 类别内 NMS → 画框。**会话复用**，不每帧重复加载模型。

**性能优化手段**：release 配置开 `opt-level=3`、`lto=true`、`codegen-units=1`；`ort` 设置图优化级别 Level3、限制 intra 线程数。

**结果**：单图 8 目标，`pre=9.9ms / infer=11.8ms / post=0.2ms`；基准测试（5 轮 × 40 图 = 200 次）平均 `pre=7.09ms / infer=8.84ms / post=0.27ms`，合计 **16.20ms ≈ 61.7 FPS**——相比 Python 版约 **1.5×**。

### 阶段四 · 服务化（FastAPI）

**思路**：Web 和桌面都不该各自带一套推理。集中到服务：

- `POST /api/detect`：图片上传 → JSON。
- `WS /ws/detect`：客户端持续推 JPEG 二进制帧 → 逐帧返回 JSON（视频流用 WebSocket 而非 HTTP，避免每帧握手）。
- 模型在 `startup` 只加载一次；推理是 CPU 密集操作，用 `asyncio.to_thread` 丢进线程池，**不阻塞事件循环**；开 CORS 方便前端开发调试。

**结果**：热态单帧 ≈ 15ms，WebSocket 单帧 ≈ 12ms，均正确检出 9 个目标。

### 阶段五 · 接入应用（Web + 桌面）

**Web（Vue3 + Vite）**

- 前端逻辑本质是"抓帧 → 发送 → 画框"：`getUserMedia(640×480)` → `canvas.toBlob(JPEG 0.7)` → WebSocket 发 ArrayBuffer → 收到结果按 `object-fit: contain` 比例换算出画布坐标再画框。
- 同时支持图片上传模式与摄像头实时模式。
- Vite 设 `base:'./'`，为后面 Electron 用 `file://` 加载产物埋好伏笔。

**桌面（Electron）**

- 复用同一份 `web/dist`，零前端改动。
- 主进程启动时先 `healthCheck()`；后端没起来才用 `child_process` `spawn` uvicorn，并轮询等待就绪；退出时 kill 子进程。
- 生产模式 `loadFile(web/dist/index.html)`，开发模式连 Vite dev server。

**结果**：Electron 33.4.11 成功拉起窗口，标题"YOLO 实时目标检测"，自动识别到已运行的后端。

---

## 3. 关键决策记录（Decision Log）

| # | 决策 | 备选 | 取舍理由 |
| --- | --- | --- | --- |
| D1 | 数据集用 COCO128 | 自定义数据集 / COCO 全量 | 目标是跑通流程；128 图 CPU 分钟级可训完 |
| D2 | 模型用 YOLO11n | YOLOv8n / 更大模型 | 最新、最小、CPU 最快 |
| D3 | 用 ONNX 做中间格式 | 直接用 .pt / TorchScript | 唯一跨 Python/Rust 都支持的通用格式 |
| D4 | 固定输入尺寸导出 | 动态尺寸 | 实时场景尺寸固定，固定能触发更多图优化 |
| D5 | Rust 用 `ort` | 自研推理 / 绑 C++ OpenCV | 官方绑定、性能与 C++ 一致，避免重造轮子 |
| D6 | 实时通信用 WebSocket | 每帧 HTTP | 高频小包，长连接省握手开销 |
| D7 | 桌面用 Electron | Tauri | 前端已就绪，Electron 零改动复用 `web/dist`，还能顺手拉起 Python 后端 |
| D8 | 推理集中在 FastAPI 服务 | 各端各自推理 | 模型只加载一次、逻辑只维护一份 |

---

## 4. 踩坑与复盘

工程中真正耗时的是环境与工具链问题，记录下来供复用。

### 4.1 Rust 工具链 / 交叉编译

- **现象**：编译 `windows-sys` 时报 `dlltool could not create import library`（CreateProcess 失败）。
- **根因**：`stable-x86_64-pc-windows-gnu` 的 self-contained 目录缺 `as.exe`，而 `dlltool` 需要它。
- **解决**：从 MSYS2 镜像取 `binutils`，把 `as.exe` 及相关 dll 放进该 self-contained 目录。

### 4.2 `ort` 接入的版本与类型坑

- **ndarray 版本冲突**：项目锁 `0.16`，而 `ort 2.0.0-rc.13` 依赖 `0.17`，导致"multiple versions of crate"错误。→ 统一为 **0.17**。
- **错误类型不满足 `Sync`**：`ort::Error<SessionBuilder>` 里含裸指针，无法被 `anyhow` 自动 `?` 转换。→ 写一个 `oerr()` 辅助函数，把任意 `Display` 错误统一转成 `anyhow::Error`。
- **`session.run` 需要 `&mut self`**：会话要可变借用，据此把 `infer` 与所有调用点改成 `&mut`。
- **`ort` 默认 feature 太重**：默认会拉 `ureq`/`native-tls`/`schannel`。→ 关掉 default features，只留 `load-dynamic` + `ndarray`，运行时用 `ORT_DYLIB_PATH` 指向 venv 内的 `onnxruntime.dll`。

### 4.3 环境与沙箱限制

- **禁止写工作区外**：Ultralytics 默认把配置/字体写进用户目录 → 用 `YOLO_CONFIG_DIR` 重定向到工作区内的 `.ultralytics/`。
- **Electron 二进制装不上**：默认缓存目录在 AppData，被沙箱拦截；且 `install.js` 读的是**小写**环境变量 `electron_config_cache`。→ 用它把缓存指向工作区后可下载；`extract-zip` 解压不完整，改用 `tar.exe` 手动解压并写 `path.txt`。
- **Electron 启动即退**：首次直接运行会静默退出，加 `--no-sandbox` 并把 `--user-data-dir` 指向工作区内后正常运行。
- **PowerShell 执行策略**：`.ps1` 被禁用导致 `npm.ps1` 无法执行 → 改用 `npm.cmd`。

### 4.4 其他

- **CLI 参数位置**：`--out` 是**全局参数**，必须写在子命令之前（`yolo-infer --out ... image x.jpg`），否则报 unexpected argument。

---

## 5. 性能与结果汇总

| 指标 | Python（ultralytics） | Rust（ort） | 后端（FastAPI 热态） |
| --- | --- | --- | --- |
| 单帧耗时 | ≈ 24ms（视频均值） | ≈ 16.2ms | ≈ 12–15ms |
| 吞吐 | ≈ 41 FPS | ≈ 61.7 FPS | — |
| 备注 | 快速验证版 | LTO/opt3 释放版 | 线程池执行，不阻塞事件循环 |

训练：COCO128 / YOLO11n / 30 epochs / CPU → **mAP50 = 0.7111，mAP50-95 = 0.5529**。

---

## 6. 后续可优化

- 换更大数据集 + 数据增强调参，提升泛化。
- 启用 INT8 量化 / OpenVINO / DirectML 进一步提升 CPU 吞吐。
- 服务端支持批处理与多实例，应对高并发。
- 桌面端做打包（electron-builder），并内嵌 Python 运行时做到真正"开箱即用"。
