import { useEffect, useRef, useState, useCallback } from 'react'

// ort is loaded via CDN script tag in index.html
/* global ort */

// ─── Model constants ───────────────────────────────────────────────────────
const MODEL_PATH = import.meta.env.BASE_URL + 'best.onnx'
const INPUT_SIZE  = 640

// ─── Letterbox: resize+pad image to INPUT_SIZE×INPUT_SIZE ─────────────────
function letterbox(img) {
  const scale = Math.min(INPUT_SIZE / img.width, INPUT_SIZE / img.height)
  const nw    = Math.round(img.width  * scale)
  const nh    = Math.round(img.height * scale)
  const padX  = Math.round((INPUT_SIZE - nw) / 2)
  const padY  = Math.round((INPUT_SIZE - nh) / 2)

  const cvs = document.createElement('canvas')
  cvs.width = INPUT_SIZE
  cvs.height = INPUT_SIZE
  const ctx = cvs.getContext('2d')
  ctx.fillStyle = '#808080'
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE)
  ctx.drawImage(img, padX, padY, nw, nh)
  return { cvs, scale, padX, padY }
}

// ─── Extract CHW float32 tensor from canvas ────────────────────────────────
function canvasToTensor(cvs) {
  const ctx  = cvs.getContext('2d')
  const data = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data
  const tensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE)
  for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
    tensor[i]                          = data[i * 4]     / 255 // R
    tensor[INPUT_SIZE * INPUT_SIZE + i] = data[i * 4 + 1] / 255 // G
    tensor[2 * INPUT_SIZE * INPUT_SIZE + i] = data[i * 4 + 2] / 255 // B
  }
  return new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE])
}

// ─── IoU for NMS ──────────────────────────────────────────────────────────
function iou(a, b) {
  const x1 = Math.max(a[0], b[0])
  const y1 = Math.max(a[1], b[1])
  const x2 = Math.min(a[2], b[2])
  const y2 = Math.min(a[3], b[3])
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const aArea = (a[2] - a[0]) * (a[3] - a[1])
  const bArea = (b[2] - b[0]) * (b[3] - b[1])
  return inter / (aArea + bArea - inter + 1e-6)
}

// ─── Parse model output → filtered boxes [{cx, cy}] ──────────────────────
// output0: [1, 37, 8400]  (cx, cy, w, h, conf, 32 mask coeffs)
function parseDetections(output0, confThresh, iouThresh, scale, padX, padY) {
  const data = output0.data          // Float32Array
  const numDet = 8400
  const boxes = []

  for (let i = 0; i < numDet; i++) {
    const conf = data[4 * numDet + i]
    if (conf < confThresh) continue

    const cx640 = data[0 * numDet + i]
    const cy640 = data[1 * numDet + i]
    const w640  = data[2 * numDet + i]
    const h640  = data[3 * numDet + i]

    // Convert from letterboxed 640 space to original image coords
    const cx = (cx640 - padX) / scale
    const cy = (cy640 - padY) / scale
    const w  = w640 / scale
    const h  = h640 / scale

    boxes.push({ cx, cy, conf, x1: cx - w/2, y1: cy - h/2, x2: cx + w/2, y2: cy + h/2 })
  }

  // Sort by confidence descending
  boxes.sort((a, b) => b.conf - a.conf)

  // NMS
  const keep = []
  const suppressed = new Uint8Array(boxes.length)
  for (let i = 0; i < boxes.length; i++) {
    if (suppressed[i]) continue
    keep.push(boxes[i])
    for (let j = i + 1; j < boxes.length; j++) {
      if (!suppressed[j] && iou(
        [boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2],
        [boxes[j].x1, boxes[j].y1, boxes[j].x2, boxes[j].y2]
      ) > iouThresh) {
        suppressed[j] = 1
      }
    }
  }

  return keep
}

// ─── Draw image + green dots on canvas ────────────────────────────────────
function drawResults(canvas, img, detections, dotRadius) {
  canvas.width  = img.width
  canvas.height = img.height
  const ctx = canvas.getContext('2d')
  ctx.drawImage(img, 0, 0)

  for (const { cx, cy } of detections) {
    ctx.beginPath()
    ctx.arc(cx, cy, dotRadius, 0, Math.PI * 2)
    ctx.fillStyle   = 'rgba(34, 197, 94, 0.85)'
    ctx.fill()
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth   = Math.max(1.5, dotRadius * 0.3)
    ctx.stroke()
  }
}

// ─── App ──────────────────────────────────────────────────────────────────
export default function App() {
  const sessionRef   = useRef(null)
  const lastImgRef   = useRef(null)
  const lastDetsRef  = useRef([])
  const canvasRef    = useRef(null)
  const wrapRef      = useRef(null)
  const dragRef      = useRef(null)   // {startX, startY, panX, panY} while dragging

  const [status,     setStatus]     = useState({ type: 'loading', text: 'Loading YOLO model… (first visit ~39 MB, then cached)' })
  const [count,      setCount]      = useState(null)
  const [dragover,   setDragover]   = useState(false)
  const [confThresh, setConfThresh] = useState(0.20)
  const [iouThresh,  setIouThresh]  = useState(0.40)
  const [dotRadius,  setDotRadius]  = useState(8)
  const [zoom,       setZoom]       = useState(1)
  const [pan,        setPan]        = useState({ x: 0, y: 0 })

  // Scroll-to-zoom (non-passive so we can preventDefault)
  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const onWheel = (e) => {
      e.preventDefault()
      const rect = wrap.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
      setZoom(prev => {
        const newZoom = Math.min(Math.max(prev * factor, 0.5), 12)
        setPan(p => ({
          x: mx - (mx - p.x) * (newZoom / prev),
          y: my - (my - p.y) * (newZoom / prev),
        }))
        return newZoom
      })
    }
    wrap.addEventListener('wheel', onWheel, { passive: false })
    return () => wrap.removeEventListener('wheel', onWheel)
  }, [])

  // Drag-to-pan handlers
  const onMouseDown = (e) => {
    if (zoom <= 1) return
    dragRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y }
    e.currentTarget.style.cursor = 'grabbing'
  }
  const onMouseMove = (e) => {
    if (!dragRef.current) return
    setPan({
      x: dragRef.current.panX + (e.clientX - dragRef.current.startX),
      y: dragRef.current.panY + (e.clientY - dragRef.current.startY),
    })
  }
  const onMouseUp = (e) => {
    dragRef.current = null
    e.currentTarget.style.cursor = zoom > 1 ? 'grab' : 'default'
  }

  const resetZoom = () => { setZoom(1); setPan({ x: 0, y: 0 }) }

  // Load model on mount
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const session = await ort.InferenceSession.create(MODEL_PATH, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        })
        if (cancelled) return
        sessionRef.current = session
        setStatus({ type: 'ready', text: 'Model loaded — drop or select an egg image.' })
      } catch (e) {
        if (!cancelled) setStatus({ type: 'error', text: 'Failed to load model: ' + e.message })
      }
    })()
    return () => { cancelled = true }
  }, [])

  const runInference = useCallback(async (img, conf, iou_, radius) => {
    if (!sessionRef.current) return
    setStatus({ type: 'running', text: 'Running inference…' })
    try {
      const { cvs, scale, padX, padY } = letterbox(img)
      const tensor = canvasToTensor(cvs)
      const feeds  = { [sessionRef.current.inputNames[0]]: tensor }
      const results = await sessionRef.current.run(feeds)
      const output0 = results[sessionRef.current.outputNames[0]]

      const dets = parseDetections(output0, conf, iou_, scale, padX, padY)
      lastDetsRef.current = dets
      drawResults(canvasRef.current, img, dets, radius)

      setCount(dets.length)
      setStatus({ type: 'done', text: `Done — found ${dets.length} egg${dets.length !== 1 ? 's' : ''}.` })
    } catch (e) {
      setStatus({ type: 'error', text: 'Inference error: ' + e.message })
    }
  }, [])

  const handleImage = useCallback((img) => {
    lastImgRef.current = img
    runInference(img, confThresh, iouThresh, dotRadius)
  }, [confThresh, iouThresh, dotRadius, runInference])

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    if (!sessionRef.current) {
      setStatus({ type: 'error', text: 'Model is still loading, please wait…' })
      return
    }
    const reader = new FileReader()
    reader.onload = (ev) => {
      const img = new Image()
      img.onload = () => handleImage(img)
      img.src = ev.target.result
    }
    reader.readAsDataURL(file)
  }, [handleImage])

  // Re-run inference when conf/iou change
  useEffect(() => {
    if (lastImgRef.current && sessionRef.current) {
      runInference(lastImgRef.current, confThresh, iouThresh, dotRadius)
    }
  }, [confThresh, iouThresh, runInference])  // eslint-disable-line react-hooks/exhaustive-deps

  // Just redraw (no inference) when dot radius changes
  useEffect(() => {
    if (lastImgRef.current && lastDetsRef.current.length > 0) {
      drawResults(canvasRef.current, lastImgRef.current, lastDetsRef.current, dotRadius)
    }
  }, [dotRadius])

  const handleReset = () => {
    lastImgRef.current = null
    lastDetsRef.current = []
    setCount(null)
    resetZoom()
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      canvasRef.current.width = 0
    }
    setStatus({ type: 'ready', text: 'Model loaded — drop or select an egg image.' })
  }

  const hasResult = count !== null

  return (
    <div className="app">
      <header>
        <h1>EggNum</h1>
        <p>Upload an image — YOLO runs <strong>entirely in your browser</strong>, no data leaves your device.</p>
      </header>

      <div className="card">
        {!hasResult && (
          <div
            className={`drop-zone${dragover ? ' dragover' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragover(true) }}
            onDragLeave={() => setDragover(false)}
            onDrop={(e) => { e.preventDefault(); setDragover(false); handleFile(e.dataTransfer.files[0]) }}
          >
            <input
              type="file"
              accept="image/*"
              onChange={(e) => handleFile(e.target.files[0])}
            />
            <span className="icon">📂</span>
            <p><strong>Click to choose</strong> or drag &amp; drop an image here</p>
            <p style={{ marginTop: '0.3rem' }}>PNG, JPG, JPEG, WebP supported</p>
          </div>
        )}

        <div className="controls">
          <div className="control-group">
            <label>
              Confidence threshold <span>{confThresh.toFixed(2)}</span>
            </label>
            <input
              type="range" min="0.05" max="0.95" step="0.05"
              value={confThresh}
              onChange={(e) => setConfThresh(+e.target.value)}
            />
          </div>
          <div className="control-group">
            <label>
              IoU threshold <span>{iouThresh.toFixed(2)}</span>
            </label>
            <input
              type="range" min="0.05" max="0.95" step="0.05"
              value={iouThresh}
              onChange={(e) => setIouThresh(+e.target.value)}
            />
          </div>
          <div className="control-group">
            <label>
              Dot size <span>{dotRadius}px</span>
            </label>
            <input
              type="range" min="3" max="30" step="1"
              value={dotRadius}
              onChange={(e) => setDotRadius(+e.target.value)}
            />
          </div>
        </div>

        <div className="status-bar">
          {(status.type === 'loading' || status.type === 'running') && (
            <div className="spinner" />
          )}
          <span>{status.text}</span>
        </div>

        {hasResult && (
          <div className="count-display">
            <div className="label">Eggs Detected</div>
            <div className="number">{count}</div>
            <div className="sub">{count === 1 ? '1 egg found' : `${count} eggs found`}</div>
          </div>
        )}

        <div
          ref={wrapRef}
          className="canvas-wrap"
          style={{ display: hasResult ? 'block' : 'none', cursor: zoom > 1 ? 'grab' : 'default' }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        >
          <canvas
            ref={canvasRef}
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: '0 0',
            }}
          />
          {zoom !== 1 && (
            <button className="zoom-reset-btn" onClick={resetZoom}>
              Reset zoom ({Math.round(zoom * 100)}%)
            </button>
          )}
        </div>

        {hasResult && (
          <button className="reset-btn" onClick={handleReset}>
            Upload another image
          </button>
        )}
      </div>

      <footer>
        Powered by <a href="https://onnxruntime.ai/" target="_blank" rel="noreferrer">ONNX Runtime Web</a> &amp;{' '}
        <a href="https://ultralytics.com/yolo" target="_blank" rel="noreferrer">YOLOv11</a>
      </footer>
    </div>
  )
}
