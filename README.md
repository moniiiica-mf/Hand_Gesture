# ASL Hand Gesture Recognition

Browser-based American Sign Language (ASL) recognition system with two modes:

- **Letter mode** — fingerspelling (A–Y + digits + J/Z motion) via an embedded Random Forest, runs fully offline in the browser.
- **Word mode** — dynamic word-level signs ("hello", "how", "how are you", "you", "today") via a GRU trained on MS-ASL clips.

---

## Quick start (letter mode only — no training required)

```bash
# Serve the app from the project root
python -m http.server 8080
# Open http://localhost:8080 in a browser and allow camera access.
```

The letter recognizer is fully self-contained in `index.html`.

---

## Word-mode pipeline (Steps B → H)

### Prerequisites

```bash
pip install mediapipe opencv-python numpy torch scikit-learn matplotlib
pip install yt-dlp          # for clip downloads
# ffmpeg and ffprobe must be on PATH
```

### Step B — Build manifest

```bash
python data/prepare_manifest.py \
    --msasl-dir MS-ASL \
    --out data/processed/msasl_5word_manifest.csv
```

Outputs:
- `data/processed/msasl_5word_manifest.csv` (241 clips)
- `data/processed/label_to_id.json`
- `data/processed/id_to_label.json`

### Step C — Download clips

```bash
python data/download_clips.py \
    --manifest data/processed/msasl_5word_manifest.csv \
    --out-dir  data/raw/msasl_5word \
    --workers  4
```

- Downloads + trims YouTube clips via yt-dlp + ffmpeg.
- Idempotent: skips already-downloaded files.
- If the dead-link rate exceeds 50%, prints local recording instructions.
- Report: `logs/download_report.json`

### Step D — Extract landmark sequences

```bash
python features/extract_sequences.py \
    --raw-dir data/raw/msasl_5word \
    --out-dir data/processed \
    --seq-len 30
```

Outputs per split:
- `data/processed/X_{train,val,test}.npy`  — shape `(N, 30, 63)`
- `data/processed/y_{train,val,test}.npy`  — shape `(N,)`
- `data/processed/extraction_stats.json`

Feature layout: 21 MediaPipe hand landmarks × 3 (x, y, z), wrist-centred and palm-width-scaled.

### Step E — Train GRU classifier

```bash
python train/train_model.py \
    --data-dir data/processed \
    --ckpt-dir train/checkpoints \
    --epochs   60 \
    --hidden   128
```

Outputs:
- `train/checkpoints/best_model.pt`
- `train/checkpoints/metrics.json`
- `train/checkpoints/confusion_matrix.png`
- `train/checkpoints/train_history.json`

### Step F — Export model to TF.js

```bash
pip install onnx onnx2tf tensorflowjs

python inference/export_tfjs.py \
    --ckpt train/checkpoints/best_model.pt \
    --out  model/word_gru
```

Places `model/word_gru/model.json` (+ weight shards) where `index.html` expects them.

### Step G — Run the app in word mode

```bash
python -m http.server 8080
```

Open `http://localhost:8080`, click **Switch to Word Mode**.  The badge shows the predicted word; the phrase bar assembles the sentence.

---

## Project layout

```
Hand_Gesture/
├── index.html                 # Browser app (letter + word mode)
├── check_connections.py       # Verify camera + library setup
├── data/
│   ├── prepare_manifest.py    # Step B
│   ├── download_clips.py      # Step C
│   ├── processed/             # Manifests, .npy arrays, label maps
│   └── raw/msasl_5word/       # Downloaded video clips
├── features/
│   └── extract_sequences.py   # Step D
├── train/
│   ├── train_model.py         # Step E
│   └── checkpoints/           # best_model.pt, metrics, plots
├── inference/
│   ├── phrase_assembler.py    # Step G — phrase decoder
│   └── export_tfjs.py         # Step F helper — PT → TF.js
├── model/
│   └── word_gru/              # TF.js model files (after export)
├── MS-ASL/                    # MS-ASL metadata JSONs
└── logs/
    └── download_report.json
```

---

## Adding new signs

1. Add the new gloss to `VOCAB` in `data/prepare_manifest.py`.
2. Re-run Steps B → E.
3. Re-export (Step F) and reload `index.html`.
4. Update `WORD_CLASSES` in `index.html` to match the new class list.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `yt-dlp not found` | `pip install yt-dlp` or `brew install yt-dlp` |
| `ffmpeg not found` | Install from ffmpeg.org or `apt install ffmpeg` |
| High dead-link rate | Follow fallback recording instructions printed by Step C |
| Word model not loading | Run Steps E and F; check browser console for CORS errors |
| Low detection rate in extraction stats | Check video quality; MediaPipe needs a clear hand view |
