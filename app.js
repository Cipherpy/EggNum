/* ─────────────────────────────────────────────
   EggNum – YOLO segmentation egg counter
   Runs entirely client-side via ONNX Runtime Web
   Model: best.onnx  (1 class: egg, YOLOv11-seg)
   Input:  [1, 3, 640, 640]
   Output0:[1, 37, 8400]  → cx,cy,w,h, conf, 32 mask coeffs
   Output1:[1, 32, 160, 160] prototype masks (unused for counting)
───────────────────────────────────────────── */

const MODEL_PATH   = 'best.onnx';
const INPUT_SIZE   = 640;
const CONF_DEFAULT = 0.25;
const IOU_DEFAULT  = 0.45;
const BOX_COLOR    = '#facc15';
const BOX_WIDTH    = 2;

// ── DOM refs ──────────────────────────────────
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const statusBar  = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const countDisp  = document.getElementById('count-display');
const eggCount   = document.getElementById('egg-count');
const eggSub     = document.getElementById('egg-sub');
const canvasWrap = document.getElementById('canvas-wrap');
const canvas     = document.getElementById('result-canvas');
const ctx        = canvas.getContext('2d');
const resetBtn   = document.getElementById('reset-btn');
const confSlider = document.getElementById('conf-slider');
const iouSlider  = document.getElementById('iou-slider');
const confVal    = document.getElementById('conf-val');
const iouVal     = document.getElementById('iou-val');

let session = null;           // ONNX session
let lastImage = null;         // HTMLImageElement of last upload

// ── Slider labels ─────────────────────────────
confSlider.addEventListener('input', () => { confVal.textContent = (+confSlider.value).toFixed(2); rerunIfReady(); });
iouSlider.addEventListener('input',  () => { iouVal.textContent  = (+iouSlider.value).toFixed(2);  rerunIfReady(); });

function rerunIfReady() {
  if (lastImage && session) runInference(lastImage);
}

// ── Model loading ─────────────────────────────
async function loadModel() {
  setStatus('loading', 'Loading YOLO model… (first visit downloads ~39 MB, then cached)');
  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    setStatus('ready', 'Model loaded — drop or select an egg image to count.');
  } catch (e) {
    setStatus('error', 'Failed to load model: ' + e.message);
    console.error(e);
  }
}

// ── File handling ─────────────────────────────
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file) {
  if (!session) { setStatus('error', 'Model is still loading, please wait…'); return; }
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => { lastImage = img; runInference(img); };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

// ── Preprocessing ─────────────────────────────
// Letterbox resize: fit image into 640×640 with grey padding
function letterbox(img) {
  const scale = Math.min(INPUT_SIZE / img.width, INPUT_SIZE / img.height);
  const nw = Math.round(img.width  * scale);
  const nh = Math.round(img.height * scale);
  const padX = Math.round((INPUT_SIZE - nw) / 2);
  const padY = Math.round((INPUT_SIZE - nh) / 2);

  const offCvs = document.createElement('canvas');
  offCvs.width = offCvs.height = INPUT_SIZE;
  const offCtx = offCvs.getContext('2d');
  offCtx.fillStyle = '#808080';
  offCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  offCtx.drawImage(img, padX, padY, nw, nh);

  return { canvas: offCvs, scale, padX, padY };
}

function canvasToTensor(cvs) {
  const { data } = cvs.getContext('2d').getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = INPUT_SIZE * INPUT_SIZE;
  const float32 = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) {
    float32[i]             = data[i * 4]     / 255; // R
    float32[pixels + i]    = data[i * 4 + 1] / 255; // G
    float32[2 * pixels + i]= data[i * 4 + 2] / 255; // B
  }
  return new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

// ── Postprocessing ────────────────────────────
// output0: [1, 37, 8400] → row-major in Float32Array
// feature layout per anchor: [cx, cy, w, h, cls_conf, mask×32]
function parseDetections(data, numAnchors, confThreshold) {
  const boxes = [];
  for (let i = 0; i < numAnchors; i++) {
    const conf = data[4 * numAnchors + i];          // single class conf
    if (conf < confThreshold) continue;
    const cx = data[0 * numAnchors + i];
    const cy = data[1 * numAnchors + i];
    const w  = data[2 * numAnchors + i];
    const h  = data[3 * numAnchors + i];
    boxes.push({ cx, cy, w, h, conf });
  }
  return boxes;
}

function iou(a, b) {
  const ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
  const ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
  const bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

function nms(boxes, iouThreshold) {
  boxes.sort((a, b) => b.conf - a.conf);
  const keep = [];
  const suppressed = new Uint8Array(boxes.length);
  for (let i = 0; i < boxes.length; i++) {
    if (suppressed[i]) continue;
    keep.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (!suppressed[j] && iou(boxes[i], boxes[j]) > iouThreshold)
        suppressed[j] = 1;
    }
  }
  return keep;
}

// Convert 640-space box back to original image space
function rescaleBox(box, scale, padX, padY, imgW, imgH) {
  const x1 = Math.max(0, (box.cx - box.w / 2 - padX) / scale);
  const y1 = Math.max(0, (box.cy - box.h / 2 - padY) / scale);
  const x2 = Math.min(imgW, (box.cx + box.w / 2 - padX) / scale);
  const y2 = Math.min(imgH, (box.cy + box.h / 2 - padY) / scale);
  return { x1, y1, x2, y2, conf: box.conf };
}

// ── Inference ─────────────────────────────────
async function runInference(img) {
  setStatus('loading', 'Running YOLO inference…');
  countDisp.style.display  = 'none';
  canvasWrap.style.display = 'none';
  resetBtn.style.display   = 'none';

  try {
    const { canvas: lbCvs, scale, padX, padY } = letterbox(img);
    const tensor = canvasToTensor(lbCvs);

    const results = await session.run({ images: tensor });
    const out0    = results['output0'].data;       // Float32Array [1,37,8400]
    const numAnchors = 8400;
    const confThreshold = +confSlider.value;
    const iouThreshold  = +iouSlider.value;

    const raw   = parseDetections(out0, numAnchors, confThreshold);
    const final = nms(raw, iouThreshold);
    const boxes = final.map(b => rescaleBox(b, scale, padX, padY, img.width, img.height));

    drawResults(img, boxes);

    const count = boxes.length;
    eggCount.textContent = count;
    eggSub.textContent   = count === 1
      ? '1 egg found'
      : `${count} eggs found at conf ≥ ${confThreshold.toFixed(2)}`;
    countDisp.style.display  = 'block';
    canvasWrap.style.display = 'block';
    resetBtn.style.display   = 'block';
    setStatus('ready', `Done — ${count} egg${count !== 1 ? 's' : ''} detected.`);
  } catch (e) {
    setStatus('error', 'Inference error: ' + e.message);
    console.error(e);
  }
}

// ── Drawing ───────────────────────────────────
function drawResults(img, boxes) {
  canvas.width  = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  // Draw bounding boxes
  ctx.strokeStyle = BOX_COLOR;
  ctx.lineWidth   = Math.max(BOX_WIDTH, img.width / 400);
  ctx.fillStyle   = BOX_COLOR;
  ctx.font        = `bold ${Math.max(12, img.width / 60)}px sans-serif`;
  ctx.textBaseline = 'top';

  boxes.forEach((b, idx) => {
    const x = b.x1, y = b.y1, w = b.x2 - b.x1, h = b.y2 - b.y1;
    // box
    ctx.strokeRect(x, y, w, h);
    // label background
    const label = `${idx + 1} · ${(b.conf * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width + 6;
    const th = parseInt(ctx.font) + 4;
    const ly = y > th ? y - th : y;
    ctx.globalAlpha = 0.75;
    ctx.fillStyle = '#78350f';
    ctx.fillRect(x, ly, tw, th);
    ctx.globalAlpha = 1;
    ctx.fillStyle = BOX_COLOR;
    ctx.fillText(label, x + 3, ly + 2);
  });
}

// ── Helpers ───────────────────────────────────
function setStatus(state, msg) {
  statusBar.className = state === 'loading' ? 'loading' : '';
  statusText.textContent = msg;
}

resetBtn.addEventListener('click', () => {
  lastImage = null;
  countDisp.style.display  = 'none';
  canvasWrap.style.display = 'none';
  resetBtn.style.display   = 'none';
  fileInput.value = '';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  setStatus('ready', 'Model ready — drop or select an egg image to count.');
});

// ── Bootstrap ─────────────────────────────────
loadModel();
