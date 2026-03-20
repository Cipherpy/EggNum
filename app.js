/* ─────────────────────────────────────────────
   EggNum – YOLO segmentation egg counter
   Runs entirely client-side via ONNX Runtime Web
   Model: best.onnx  (1 class: egg, YOLOv11-seg)
   Input:  [1, 3, 640, 640]
   Output0:[1, 37, 8400]  → cx,cy,w,h, conf, 32 mask coeffs
   Output1:[1, 32, 160, 160] prototype masks
───────────────────────────────────────────── */

const MODEL_PATH = 'best.onnx';
const INPUT_SIZE  = 640;
const PROTO_SIZE  = 160;           // prototype mask resolution
const MASK_THRESH = 0.5;           // sigmoid threshold for mask pixels

// Per-egg colours (RGBA fill + stroke)
const PALETTE = [
  [250, 204,  21], // amber
  [249, 115,  22], // orange
  [ 34, 197,  94], // green
  [ 59, 130, 246], // blue
  [168,  85, 247], // purple
  [236,  72, 153], // pink
  [ 14, 165, 233], // sky
  [ 20, 184, 166], // teal
  [239,  68,  68], // red
  [132, 204,  22], // lime
];

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

let session   = null;
let lastImage = null;

// ── Threshold sliders ─────────────────────────
confSlider.addEventListener('input', () => { confVal.textContent = (+confSlider.value).toFixed(2); rerunIfReady(); });
iouSlider.addEventListener('input',  () => { iouVal.textContent  = (+iouSlider.value).toFixed(2);  rerunIfReady(); });
function rerunIfReady() { if (lastImage && session) runInference(lastImage); }

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
  const pixels  = INPUT_SIZE * INPUT_SIZE;
  const float32 = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) {
    float32[i]              = data[i * 4]     / 255;
    float32[pixels + i]     = data[i * 4 + 1] / 255;
    float32[2 * pixels + i] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

// ── Detection parsing ─────────────────────────
// output0: Float32Array shaped [1, 37, 8400] (row-major)
// Per anchor: [cx, cy, w, h, cls_conf, mask_coeff×32]
function parseDetections(data, numAnchors, confThreshold) {
  const dets = [];
  for (let i = 0; i < numAnchors; i++) {
    const conf = data[4 * numAnchors + i];
    if (conf < confThreshold) continue;
    const cx = data[0 * numAnchors + i];
    const cy = data[1 * numAnchors + i];
    const w  = data[2 * numAnchors + i];
    const h  = data[3 * numAnchors + i];
    const coeffs = new Float32Array(32);
    for (let c = 0; c < 32; c++) {
      coeffs[c] = data[(5 + c) * numAnchors + i];
    }
    dets.push({ cx, cy, w, h, conf, coeffs });
  }
  return dets;
}

// ── NMS ───────────────────────────────────────
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

function nms(dets, iouThreshold) {
  dets.sort((a, b) => b.conf - a.conf);
  const keep = [];
  const suppressed = new Uint8Array(dets.length);
  for (let i = 0; i < dets.length; i++) {
    if (suppressed[i]) continue;
    keep.push(dets[i]);
    for (let j = i + 1; j < dets.length; j++) {
      if (!suppressed[j] && iou(dets[i], dets[j]) > iouThreshold)
        suppressed[j] = 1;
    }
  }
  return keep;
}

// ── Segmentation mask computation ─────────────
// For each detection compute sigmoid(coeffs · prototypes) → 160×160 mask
// protData: Float32Array [32 × 160 × 160] (from output1[0])
function computeMask(coeffs, protData) {
  const maskSize = PROTO_SIZE * PROTO_SIZE; // 25600
  const mask = new Float32Array(maskSize);
  for (let p = 0; p < maskSize; p++) {
    let val = 0;
    for (let c = 0; c < 32; c++) {
      val += coeffs[c] * protData[c * maskSize + p];
    }
    mask[p] = 1 / (1 + Math.exp(-val)); // sigmoid
  }
  return mask; // values 0..1
}

// ── Inference ─────────────────────────────────
async function runInference(img) {
  setStatus('loading', 'Running YOLO inference…');
  countDisp.style.display  = 'none';
  canvasWrap.style.display = 'none';
  resetBtn.style.display   = 'none';

  try {
    const { canvas: lbCvs, scale, padX, padY } = letterbox(img);
    const tensor  = canvasToTensor(lbCvs);
    const results = await session.run({ images: tensor });

    const out0     = results['output0'].data; // [1, 37, 8400]
    const protData = results['output1'].data; // [1, 32, 160, 160]

    const confThreshold = +confSlider.value;
    const iouThreshold  = +iouSlider.value;

    const raw   = parseDetections(out0, 8400, confThreshold);
    const final = nms(raw, iouThreshold);

    // Compute segmentation mask for each surviving detection
    const masks = final.map(det => computeMask(det.coeffs, protData));

    drawResults(img, final, masks, scale, padX, padY);

    const count = final.length;
    eggCount.textContent = count;
    eggSub.textContent   = count === 1 ? '1 egg found' : `${count} eggs found`;
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
// Render each egg's segmentation mask as a semi-transparent coloured overlay.
// The 160×160 mask lives in letterboxed-640 space; we inverse-transform it
// onto the original image canvas using the letterbox parameters.
function drawResults(img, _dets, masks, scale, padX, padY) {
  canvas.width  = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  // How the 160×160 mask maps onto the original-image canvas:
  //   mask pixel (mx, my) → 640-space (mx×4, my×4)
  //   640-space → original: x = (px − padX) / scale
  //   So the whole 160×160 mask drawn at:
  //     dest x = −padX / scale,  dest y = −padY / scale
  //     dest w = INPUT_SIZE / scale,  dest h = INPUT_SIZE / scale
  const dstX = -padX / scale;
  const dstY = -padY / scale;
  const dstW =  INPUT_SIZE / scale;
  const dstH =  INPUT_SIZE / scale;

  // Shared scratch canvas for mask pixels
  const maskCvs    = document.createElement('canvas');
  maskCvs.width    = PROTO_SIZE;
  maskCvs.height   = PROTO_SIZE;
  const maskCtx    = maskCvs.getContext('2d');
  const maskPixels = maskCtx.createImageData(PROTO_SIZE, PROTO_SIZE);

  masks.forEach((mask, idx) => {
    const [r, g, b] = PALETTE[idx % PALETTE.length];

    // Fill mask pixels
    const d = maskPixels.data;
    for (let p = 0; p < mask.length; p++) {
      const on = mask[p] > MASK_THRESH;
      const i4 = p * 4;
      d[i4]     = on ? r : 0;
      d[i4 + 1] = on ? g : 0;
      d[i4 + 2] = on ? b : 0;
      d[i4 + 3] = on ? 170 : 0;   // semi-transparent fill
    }
    maskCtx.putImageData(maskPixels, 0, 0);

    // Render scaled over original image
    ctx.drawImage(maskCvs, dstX, dstY, dstW, dstH);

    // Crisp outline: draw the mask again at ~10% opacity for a soft glow border,
    // then re-scale as a stroke using canvas filter trick
    ctx.save();
    ctx.globalAlpha = 0.9;
    // Stroke outline: use composite trick – draw fill, then erode inward
    // Simpler: draw a second pass with globalCompositeOperation='source-atop'
    // to add a brighter border. We use shadowBlur instead.
    ctx.shadowColor = `rgb(${r},${g},${b})`;
    ctx.shadowBlur  = 6;
    ctx.drawImage(maskCvs, dstX, dstY, dstW, dstH);
    ctx.restore();
  });
}

// ── Helpers ───────────────────────────────────
function setStatus(state, msg) {
  statusBar.className    = state === 'loading' ? 'loading' : '';
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
