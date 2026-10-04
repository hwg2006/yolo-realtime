// 检测框绘制：把检测结果按 object-fit: contain 的缩放与偏移映射到 overlay canvas。
// 只负责画布尺寸自适应与坐标换算，不感知数据来源。

const BOX_COLOR = '#22c55e'

/**
 * 在 overlay canvas 上绘制检测框与标签。
 * @param {HTMLCanvasElement} canvas 覆盖层画布
 * @param {Array} detections 归一化后的检测结果（含 xyxy/label/conf）
 * @param {number} srcW 原图宽度
 * @param {number} srcH 原图高度
 */
export function drawBoxes(canvas, detections, srcW, srcH) {
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
  const ox = (canvas.width - srcW * scale) / 2
  const oy = (canvas.height - srcH * scale) / 2

  ctx.lineWidth = 2
  ctx.font = '14px "Segoe UI", sans-serif'
  ctx.textBaseline = 'bottom'
  for (const d of detections) {
    const [x1, y1, x2, y2] = d.xyxy
    const X = ox + x1 * scale
    const Y = oy + y1 * scale
    const W = (x2 - x1) * scale
    const H = (y2 - y1) * scale

    ctx.strokeStyle = BOX_COLOR
    ctx.strokeRect(X, Y, W, H)

    const label = `${d.label} ${(d.conf * 100).toFixed(0)}%`
    const tw = ctx.measureText(label).width + 10
    ctx.fillStyle = 'rgba(34,197,94,0.85)'
    ctx.fillRect(X, Y - 20, tw, 20)
    ctx.fillStyle = '#04120a'
    ctx.fillText(label, X + 5, Y - 3)
  }
}

/** 清空 overlay 画布。 */
export function clearOverlay(canvas) {
  const ctx = canvas?.getContext('2d')
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
}
