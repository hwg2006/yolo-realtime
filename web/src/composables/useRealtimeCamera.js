// 摄像头实时检测数据流：采集 → 按目标帧率抓帧 → WebSocket 推送 → 结果回调。
// 只负责连接与帧循环管理；绘制、状态文案由调用方通过回调决定。
import { ref } from 'vue'
import { openDetectSocket } from '../api'

const TARGET_FPS = 30
const JPEG_QUALITY = 0.7
const CAPTURE_SIZE = { width: 640, height: 480 }

/**
 * @param {object} opts
 * @param {import('vue').Ref} opts.videoEl video 元素 ref
 * @param {(res: object) => void} [opts.onResult] 收到归一化检测结果的回调
 * @param {(text: string, ok: boolean) => void} [opts.onStatus] 状态变更回调
 */
export function useRealtimeCamera({ videoEl, onResult, onStatus } = {}) {
  const running = ref(false)
  const fps = ref(0)

  let stream = null
  let ws = null
  let rafId = 0
  let inflight = false
  let lastSent = 0
  let lastFrameTime = 0
  const captureCanvas = document.createElement('canvas')

  function loop() {
    rafId = requestAnimationFrame(loop)
    const now = performance.now()
    if (now - lastSent < 1000 / TARGET_FPS) return
    if (inflight || !ws || !ws.ready) return
    const v = videoEl?.value
    if (!v || !v.videoWidth) return

    captureCanvas.width = v.videoWidth
    captureCanvas.height = v.videoHeight
    captureCanvas.getContext('2d').drawImage(v, 0, 0)

    captureCanvas.toBlob(
      (blob) => {
        if (!blob || !ws || !ws.ready) return
        inflight = true
        lastSent = performance.now()
        blob.arrayBuffer().then((buf) => ws.send(buf))
      },
      'image/jpeg',
      JPEG_QUALITY,
    )

    // FPS 滑动平均
    if (lastFrameTime) {
      const inst = 1000 / (now - lastFrameTime)
      fps.value = fps.value ? fps.value * 0.9 + inst * 0.1 : inst
    }
    lastFrameTime = now
  }

  async function start() {
    if (running.value) return
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: CAPTURE_SIZE,
        audio: false,
      })
    } catch (e) {
      onStatus?.('无法访问摄像头: ' + e.message, false)
      return
    }
    videoEl.value.srcObject = stream
    await videoEl.value.play()

    ws = openDetectSocket({
      onOpen: () => {
        onStatus?.('已连接推理服务', true)
        running.value = true
        lastFrameTime = 0
        loop()
      },
      onClose: () => {
        onStatus?.('连接已断开', false)
        stop(true)
      },
      onError: () => onStatus?.('WebSocket 错误：后端服务是否已启动？', false),
      onResult: (res) => {
        inflight = false
        onResult?.(res)
      },
    })
  }

  function stop(fromClose = false) {
    running.value = false
    cancelAnimationFrame(rafId)
    inflight = false
    if (ws && !fromClose) ws.close()
    ws = null
    if (stream) {
      stream.getTracks().forEach((t) => t.stop())
      stream = null
    }
    fps.value = 0
    if (!fromClose) onStatus?.('已停止', false)
  }

  return { running, fps, start, stop }
}
