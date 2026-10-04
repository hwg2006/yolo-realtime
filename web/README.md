# web —— Vue3 + Vite 前端

只负责 UI 组织与订阅：抓帧 → 经 WebSocket 发送 → 收到结果后画框。
与后端的通信统一封装在 `src/api.js`，数据流与绘制拆到 `src/composables/`。

| 文件 | 作用 |
| --- | --- |
| [src/App.vue](./src/App.vue) | 主界面：摄像头 / 图片检测的 UI 组织与订阅 |
| [src/api.js](./src/api.js) | 唯一通信入口（HTTP / WebSocket 地址与请求） |
| [src/composables/useRealtimeCamera.js](./src/composables/useRealtimeCamera.js) | 摄像头采集与帧循环、WebSocket 推送、FPS 统计 |
| [src/composables/useOverlayCanvas.js](./src/composables/useOverlayCanvas.js) | overlay 画布的坐标换算与检测框绘制 |
| [src/main.js](./src/main.js) | 应用入口 |
| [src/style.css](./src/style.css) | 全局样式 |

## 运行

```powershell
cd web
npm install
npm run dev      # http://127.0.0.1:5173
npm run build    # 产出 web/dist，供桌面端加载
```

- 需先启动后端服务（默认 `127.0.0.1:8000`），可用 `VITE_API_BASE` 覆盖地址
