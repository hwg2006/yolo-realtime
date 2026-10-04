<script setup>
import { onBeforeUnmount, ref } from 'vue'
import { wsUrl, detectImage, API_BASE } from './api'

const videoEl = ref(null)
const overlayEl = ref(null)
const fileInput = ref(null)
const imgEl = ref(null)

const running = ref(false)
const status = ref('未连接')
const statusOk = ref(false)
const fps = ref(0)
const inferMs = ref(0)
const dets = ref([])
const mode = ref('camera') // camera | image

let stream = null
let ws = null
let rafId = 0
let inflight = false
let lastSent = 0
let lastFrameTime = 0
let targetFps = 30

const captureCanvas = document.createElement('canvas')

function setStatus(text, ok) {
  status.value = text
  statusOk.value = ok
}

// ---------- 绘制 ----------
function drawBoxes(srcW, srcH) {
  const canvas = overlayEl.value
  if (!canvas || !srcW || !srcH) return
  const rect = canvas.parentElement.getBoundingClientRect()
  if (canvas.width !== rect.width || canvas.height !== rect.height) {
    canvas.width = rect.width
    canvas.height = rect.height
  }
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // object-fit: contain 的缩放与偏移
  const scale = Math.min(canvas.width / srcW, canvas.height / srcH)
  const dw = srcW * scale
  const dh = srcH * scale
  const ox = (canvas.width - dw) / 2
  const oy = (canvas.height - dh) / 2

  ctx.lineWidth = 2
  ctx.font = '14px "Segoe UI", sans-serif'
  ctx.textBaseline = 'bottom'
  for (const d of dets.value) {
    const [x1, y1, x2, y2] = d.xyxy
    const X = ox + x1 * scale
    const Y = oy + y1 * scale
    const W = (x2 - x1) * scale
    const H = (y2 - y1) * scale

    ctx.strokeStyle = '#22c55e'
    ctx.strokeRect(X, Y, W, H)

    const label = `${d.label} ${(d.conf * 100).toFixed(0)}%`
    const tw = ctx.measureText(label).width + 10
    ctx.fillStyle = 'rgba(34,197,94,0.85)'
    ctx.fillRect(X, Y - 20, tw, 20)
    ctx.fillStyle = '#04120a'
    ctx.fillText(label, X + 5, Y - 3)
  }
}

// ---------- 实时摄像头 ----------
function loop() {
  rafId = requestAnimationFrame(loop)
  const now = performance.now()
  if (now - lastSent < 1000 / targetFps) return
  if (inflight || !ws || ws.readyState !== WebSocket.OPEN) return
  const v = videoEl.value
  if (!v || !v.videoWidth) return

  captureCanvas.width = v.videoWidth
  captureCanvas.height = v.videoHeight
  captureCanvas.getContext('2d').drawImage(v, 0, 0)

  captureCanvas.toBlob(
    (blob) => {
      if (!blob || !ws || ws.readyState !== WebSocket.OPEN) return
      inflight = true
      lastSent = performance.now()
      blob.arrayBuffer().then((buf) => ws.send(buf))
    },
    'image/jpeg',
    0.7,
  )

  // FPS
  if (lastFrameTime) {
    const inst = 1000 / (now - lastFrameTime)
    fps.value = fps.value ? fps.value * 0.9 + inst * 0.1 : inst
  }
  lastFrameTime = now
}

async function startCamera() {
  if (running.value) return
  mode.value = 'camera'
  dets.value = []
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    })
  } catch (e) {
    setStatus('无法访问摄像头: ' + e.message, false)
    return
  }
  videoEl.value.srcObject = stream
  await videoEl.value.play()

  ws = new WebSocket(wsUrl())
  ws.binaryType = 'arraybuffer'
  ws.onopen = () => {
    setStatus('已连接推理服务', true)
    running.value = true
    lastFrameTime = 0
    loop()
  }
  ws.onclose = () => {
    setStatus('连接已断开', false)
    stopCamera(true)
  }
  ws.onerror = () => setStatus('WebSocket 错误：后端服务是否已启动？', false)
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data)
    dets.value = data.detections || []
    inferMs.value = data.infer_ms || 0
    drawBoxes(data.width, data.height)
    inflight = false
  }
}

function stopCamera(fromClose) {
  running.value = false
  cancelAnimationFrame(rafId)
  inflight = false
  if (ws && !fromClose) {
    try { ws.close() } catch { /* ignore */ }
  }
  ws = null
  if (stream) {
    stream.getTracks().forEach((t) => t.stop())
    stream = null
  }
  const ctx = overlayEl.value?.getContext('2d')
  if (ctx) ctx.clearRect(0, 0, overlayEl.value.width, overlayEl.value.height)
  fps.value = 0
  if (!fromClose) setStatus('已停止', false)
}

// ---------- 图片上传 ----------
function pickFile() {
  fileInput.value?.click()
}

async function onFile(e) {
  const file = e.target.files?.[0]
  if (!file) return
  stopCamera(false)
  mode.value = 'image'
  dets.value = []
  const url = URL.createObjectURL(file)
  await new Promise((r) => {
    imgEl.value.onload = r
    imgEl.value.src = url
  })
  setStatus('上传中…', false)
  try {
    const res = await detectImage(file)
    dets.value = res.detections || []
    inferMs.value = res.infer_ms || 0
    drawBoxes(res.width, res.height)
    setStatus(`检测完成，共 ${dets.value.length} 个目标`, true)
  } catch (err) {
    setStatus(err.message, false)
  }
}

onBeforeUnmount(() => stopCamera(false))
</script>

<template>
  <h1>YOLO 实时目标检测</h1>
  <div class="sub">
    Vue 3 + Vite 前端 · 后端推理服务 <code>{{ API_BASE }}</code>
  </div>

  <div class="toolbar">
    <button :disabled="running" @click="startCamera">开启摄像头检测</button>
    <button class="secondary" :disabled="!running" @click="stopCamera(false)">停止</button>
    <button class="secondary" @click="pickFile">上传图片检测</button>
    <input ref="fileInput" type="file" accept="image/*" hidden @change="onFile" />
    <span :class="statusOk ? 'status-ok' : 'status-err'">{{ status }}</span>
  </div>

  <div class="stats">
    <div>FPS: <b>{{ fps.toFixed(1) }}</b></div>
    <div>推理耗时: <b>{{ inferMs.toFixed(1) }} ms</b></div>
    <div>目标数: <b>{{ dets.length }}</b></div>
  </div>

  <div class="stage">
    <video
      v-show="mode === 'camera'"
      ref="videoEl"
      playsinline
      muted
    ></video>
    <img v-show="mode === 'image'" ref="imgEl" alt="preview" />
    <canvas ref="overlayEl"></canvas>
  </div>

  <div v-if="dets.length" class="det-list">
    <div v-for="(d, i) in dets" :key="i" class="det-row">
      <span><span class="tag">{{ d.label }}</span> #{{ d.cls }}</span>
      <span>置信度 {{ (d.conf * 100).toFixed(1) }}%</span>
    </div>
  </div>
</template>
