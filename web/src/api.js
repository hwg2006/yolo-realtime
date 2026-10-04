// 与后端推理服务通信的封装。
// 可通过 .env 里的 VITE_API_BASE 覆盖（例如 Electron 打包后指向本地服务）。
const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export function wsUrl() {
  return API_BASE.replace(/^http/, 'ws') + '/ws/detect'
}

export async function detectImage(file) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${API_BASE}/api/detect`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error('检测失败: ' + res.status)
  return res.json()
}

export async function health() {
  const res = await fetch(`${API_BASE}/api/health`)
  return res.json()
}

export { API_BASE }
