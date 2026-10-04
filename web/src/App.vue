<script setup>
// 主界面组件：摄像头 / 图片目标检测的 UI 组织与订阅。
// 数据流（摄像头采集、WebSocket 收发）见 ./composables/useRealtimeCamera，
// 检测框绘制见 ./composables/useOverlayCanvas，通信细节见 ./api。
import { onBeforeUnmount, ref } from 'vue'
import { detectImage, API_BASE } from './api'
import { drawBoxes, clearOverlay } from './composables/useOverlayCanvas'
import { useRealtimeCamera } from './composables/useRealtimeCamera'

const videoEl = ref(null)
const overlayEl = ref(null)
const fileInput = ref(null)
const imgEl = ref(null)

const status = ref('未连接')
const statusOk = ref(false)
const inferMs = ref(0)
const dets = ref([])
const mode = ref('camera') // camera | image

function setStatus(text, ok) {
  status.value = text
  statusOk.value = ok
}

// ---------- 实时摄像头 ----------
const camera = useRealtimeCamera({
  videoEl,
  onStatus: setStatus,
  onResult: (res) => {
    dets.value = res.detections
    inferMs.value = res.inferMs
    drawBoxes(overlayEl.value, res.detections, res.width, res.height)
  },
})
const { running, fps } = camera

function startCamera() {
  if (running.value) return
  mode.value = 'camera'
  dets.value = []
  camera.start()
}

function stopCamera(fromClose = false) {
  camera.stop(fromClose)
  clearOverlay(overlayEl.value)
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
    dets.value = res.detections
    inferMs.value = res.inferMs
    drawBoxes(overlayEl.value, res.detections, res.width, res.height)
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
