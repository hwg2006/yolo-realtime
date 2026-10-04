# desktop —— Electron 桌面端

只做"模块组装"：启动并等待 Python 后端 → 加载前端界面 → 退出时回收子进程。

| 文件 | 作用 |
| --- | --- |
| [main.js](./main.js) | 主进程：健康检查、spawn uvicorn、创建窗口、退出清理 |

## 运行

```powershell
cd desktop
npm install
npm start        # 生产：加载 ../web/dist，并自动拉起 Python 后端
npm run dev      # 开发：连接 Vite 开发服务器
```

- 需先在 `web/` 执行 `npm run build` 生成 `web/dist`
- 环境变量：
  - `YOLO_PYTHON` 指定 Python 解释器（默认 `python`）
  - `YOLO_BACKEND_HOST` / `YOLO_BACKEND_PORT` 指定后端地址（默认 `127.0.0.1:8000`）
  - `YOLO_BACKEND_MODULE` 指定 ASGI 应用路径（默认 `server.app:app`）
  - `YOLO_BACKEND_EXTERNAL=1` 表示后端已手动启动、不再自动拉起
  - `VITE_DEV_SERVER_URL` 指定后进入开发模式并加载该地址
