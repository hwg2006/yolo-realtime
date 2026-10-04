// 与后端推理服务通信的唯一入口：统一封装接口地址、请求与响应解析。
// 后端返回的消息结构在此归一化，组件只消费结构化对象，不感知原始字段。
// 可通过 .env 里的 VITE_API_BASE 覆盖（例如 Electron 打包后指向本地服务）。

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

/** 把后端原始结果归一化为前端统一结构。 */
export function normalizeResult(payload) {
  return {
    width: payload?.width || 0,
    height: payload?.height || 0,
    inferMs: payload?.infer_ms || 0,
    detections: payload?.detections || [],
  }
}

export async function detectImage(file) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${API_BASE}/api/detect`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error('检测失败: ' + res.status)
  return normalizeResult(await res.json())
}

export async function health() {
  const res = await fetch(`${API_BASE}/api/health`)
  return res.json()
}

/**
 * 打开实时检测 WebSocket，回调均收到归一化后的结果。
 * @returns {{ send: (data) => void, close: () => void, readonly ready: boolean }}
 */
export function openDetectSocket({ onOpen, onClose, onError, onResult } = {}) {
  const ws = new WebSocket(API_BASE.replace(/^http/, 'ws') + '/ws/detect')
  ws.binaryType = 'arraybuffer'
  ws.onopen = () => onOpen?.()
  ws.onclose = () => onClose?.()
  ws.onerror = () => onError?.()
  ws.onmessage = (ev) => onResult?.(normalizeResult(JSON.parse(ev.data)))
  return {
    send(data) {
      if (ws.readyState === WebSocket.OPEN) ws.send(data)
    },
    close() {
      try { ws.close() } catch { /* ignore */ }
    },
    get ready() {
      return ws.readyState === WebSocket.OPEN
    },
  }
}

export { API_BASE }
