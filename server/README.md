# server —— FastAPI 推理服务

把推理收敛为一个服务：只做"计算 + 结果推送"，Web / 桌面端只负责采集与画框。

| 接口 | 说明 |
| --- | --- |
| `GET /api/health` | 健康检查（桌面端据此判断后端是否就绪） |
| `POST /api/detect` | 上传单张图片 → JSON 检测结果 |
| `WS /ws/detect` | 客户端持续推送 JPEG 二进制帧 → 逐帧返回 JSON 结果 |

| 文件 | 作用 |
| --- | --- |
| [app.py](./app.py) | 应用组装：创建 FastAPI、挂载中间件与路由、注册模型加载钩子 |
| [routes.py](./routes.py) | HTTP / WebSocket 路由：只做请求解析与响应封装 |
| [detector.py](./detector.py) | `Detector` 封装与全局单例的加载 / 获取 |
| [schemas.py](./schemas.py) | 前后端共享的响应数据模型 |
| [config.py](./config.py) | 默认推理参数与 CORS 白名单（路径 / 权重解析见根目录 `project_config.py`） |

## 运行

```powershell
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

- 模型在 `startup` 时加载一次；CPU 推理通过 `asyncio.to_thread` 放入线程池，避免阻塞事件循环
- CORS 白名单仅放行本地开发与桌面端来源（Vite `127.0.0.1/localhost:5173` 与 Electron `file://` 的 `Origin: null`），可用 `YOLO_ALLOWED_ORIGINS`（逗号分隔）覆盖
- 依赖：`fastapi`、`uvicorn[standard]`、`ultralytics`、`opencv-python`、`python-multipart`
