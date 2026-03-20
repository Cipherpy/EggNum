# EggNum — Browser-based Egg Counter

Upload an image and count eggs instantly using a YOLOv11 segmentation model running **entirely in your browser** via ONNX Runtime Web. No data leaves your device.

## Live Demo

👉 **[Open on GitHub Pages](https://YOUR_USERNAME.github.io/EggNum/)**

## How it works

1. `best.onnx` — YOLOv11-seg model (1 class: egg) exported from `best.pt`
2. ONNX Runtime Web runs inference in WebAssembly — no backend required
3. Results are drawn on canvas; egg count displayed immediately

## Publish to GitHub Pages

```bash
git add .
git commit -m "Add EggNum webapp"
git push
```

Then go to **Settings → Pages → Source: Deploy from branch → main / root** and save.

## Local development

Just open `index.html` in a browser served from a local HTTP server (needed for WASM):

```bash
python3 -m http.server 8080
# → http://localhost:8080
```

## Model details

| Property | Value |
|---|---|
| Architecture | YOLOv11-seg |
| Input | 640 × 640 RGB |
| Class | egg (single class) |
| Format | ONNX opset 12 |
