// Electron 主进程：启动内置 Python 推理服务 + 加载前端界面。
//
// 用法:
//   npm start          # 加载 ../web/dist（需先 npm run build）
//   npm run dev        # 连接 Vite 开发服务器 http://127.0.0.1:5173
//   YOLO_BACKEND_EXTERNAL=1 则不再自动拉起后端（假设已手动启动）

const { app, BrowserWindow, dialog } = require('electron')
const { spawn } = require('node:child_process')
const path = require('node:path')
const fs = require('node:fs')
const http = require('node:http')

const ROOT = path.resolve(__dirname, '..')
const BACKEND_HOST = '127.0.0.1'
const BACKEND_PORT = 8000
const DEV_URL = process.env.VITE_DEV_SERVER_URL || 'http://127.0.0.1:5173'
const isDev = process.argv.includes('--dev') || !!process.env.VITE_DEV_SERVER_URL

let backendProc = null
let win = null

function healthCheck() {
  return new Promise((resolve) => {
    const req = http.get(
      { host: BACKEND_HOST, port: BACKEND_PORT, path: '/api/health', timeout: 800 },
      (res) => {
        res.resume()
        resolve(res.statusCode === 200)
      },
    )
    req.on('error', () => resolve(false))
    req.on('timeout', () => { req.destroy(); resolve(false) })
  })
}

async function waitForBackend(timeoutMs = 60000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    if (await healthCheck()) return true
    await new Promise((r) => setTimeout(r, 800))
  }
  return false
}

async function startBackend() {
  if (process.env.YOLO_BACKEND_EXTERNAL === '1') {
    console.log('[electron] 跳过后端启动（外部模式）')
    return
  }
  if (await healthCheck()) {
    console.log('[electron] 检测到后端已在运行')
    return
  }
  const python = process.env.YOLO_PYTHON || 'python'
  console.log('[electron] 启动 Python 推理服务…')
  backendProc = spawn(
    python,
    ['-m', 'uvicorn', 'server.app:app', '--host', BACKEND_HOST, '--port', String(BACKEND_PORT)],
    { cwd: ROOT, stdio: 'inherit', windowsHide: true },
  )
  backendProc.on('error', (e) => console.error('[electron] 后端启动失败:', e.message))

  const ok = await waitForBackend()
  if (!ok) {
    dialog.showErrorBox(
      '推理服务未就绪',
      '无法连接 Python 推理服务 (127.0.0.1:8000)。\n请确认已安装依赖并训练/导出模型后重试。',
    )
  }
}

function createWindow() {
  win = new BrowserWindow({
    width: 1120,
    height: 860,
    title: 'YOLO 实时目标检测',
    backgroundColor: '#0f1115',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const distIndex = path.join(ROOT, 'web', 'dist', 'index.html')
  if (isDev) {
    win.loadURL(DEV_URL)
    win.webContents.openDevTools({ mode: 'detach' })
  } else if (fs.existsSync(distIndex)) {
    win.loadFile(distIndex)
  } else {
    win.loadURL(DEV_URL)
  }

  win.on('closed', () => { win = null })
}

app.whenReady().then(async () => {
  await startBackend()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  if (backendProc && !backendProc.killed) {
    backendProc.kill()
  }
})
