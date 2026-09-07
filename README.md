# ASL Hand Gesture Recognition

Browser-based American Sign Language (ASL) recognition system with two modes:

- **Letter mode** — fingerspelling (A–Y + digits + J/Z motion) via an embedded Random Forest; runs in the browser with no separate letter-model files; initial loading requires the MediaPipe and TensorFlow.js CDNs.
- **Word mode** — dynamic word-level signs ("hello", "how", "how are you", "you", "today") via a GRU trained on MS-ASL clips and served as a TF.js model.

---

## Word availability in MS-ASL

| Word requested | In MSASL_classes.json | MS-ASL label | Train clips |
|---|---|---|---|
| hello | **yes** | 0 | 30 |
| how | **yes** | 59 | 32 |
| **are** | **NO — not a standalone sign** | — | — |
| you | **yes** | 68 | 35 |
| today | **yes** | 131 | 26 |

**"are" is not in MS-ASL.** The best available substitute is the compound sign **"how are you"** (MS-ASL label 274, 19 train clips). The pipeline uses this as class `how_are_you`.

**Final 5-class vocabulary:** `hello · how · how_are_you · you · today`

> **Note on `how_are_you`:** MS-ASL only has 19 training clips for this class. After dead-link attrition (typically 20–40% of YouTube URLs are dead), you may end up with 11–15 usable clips. The default `--min-clips` threshold is set to 15 to account for this. If `how_are_you` falls below threshold, lower `--min-clips` to 10 and re-run.

---

## Quick start — letter mode only (no training required)

```bash
python -m http.server 8080
# Open http://localhost:8080 — click Start camera, then allow camera access.
```

---

## Letter mode — validation & tuning playbook

### How to run

```bash
python -m http.server 8080
# Open http://localhost:8080 in Chrome/Edge
# Allow camera access when prompted
# The letter badge appears top-left; no training required
```

### Before / after this update

| Behaviour | Before | After |
|---|---|---|
| How a letter is committed | After N consecutive matching frames | After 750 ms of stable majority vote |
| Rapid-fire duplicates | Common when holding a pose | One commit per continuous hold, plus a 700 ms cooldown gate |
| Ambiguous pose (top-2 too close) | Committed anyway | Held until margin ≥ 15 % |
| H vs U/V confusion | Frequent mis-fires | Geometry check: H only commits when fingers point sideways |
| Voice feedback | Silent | Web Speech API reads each committed letter aloud (toggle with 🔊 Voice) |
| Stability progress | None | Blue fill bar shows hold progress toward commit threshold |

### Tuning thresholds (edit the `CFG` object in `index.html`)

```javascript
const CFG = {
  BUFFER_SIZE:        20,    // frames kept in rolling vote window
  STABLE_MS:         750,    // ms the top candidate must dominate before commit
  MIN_CONFIDENCE:   0.50,    // RF confidence gate (0–1); raise to 0.65 to reduce noise
  COOLDOWN_MS:       700,    // ms blocked after each commit; prevents rapid duplicates
  TOP2_MARGIN:      0.15,    // min gap between top-2 votes (fraction of total trees)
  NO_HAND_RESET_MS:  400,    // ms without a detected hand before buffer resets
  TTS_ENABLED_DEFAULT: true, // start with voice on/off
};
```

**Guidelines:**
- Noisy environment or shaky hand → increase `STABLE_MS` to 1000–1200 ms.
- Predictions commit too slowly → lower `STABLE_MS` to 500 ms.
- Too many false commits → raise `MIN_CONFIDENCE` to 0.60 and `TOP2_MARGIN` to 0.20.
- Duplicates slip through → raise `COOLDOWN_MS` to 900 ms.
- H still misfiring → open the debug panel (`?` key) and check **idxDX** vs **idxDY**;
  H only passes if `|idxDX| > |idxDY|`. If your hand geometry is unusual, the
  threshold inside `onHandResults` can be tightened from `0.55` to a higher ratio.

### Manual QA checklist

Run through these checks after any threshold change:

- [ ] **Single commit per hold** — Hold A steady for 1 s → exactly one "A" appears in
      the history bar; no rapid repeats.
- [ ] **Cooldown gate** — Sign A, immediately re-sign A without lifting your hand →
      second commit is blocked until the hand is lowered or a different stable letter is signed.
- [ ] **No-hand reset** — Drop your hand fully for 0.5 s, then sign B → B commits
      cleanly (cooldown waived after reset).
- [ ] **H vs U/V** — Point two fingers **upward** → should predict U or V, not H.
      Point two fingers **sideways** → should predict H.
- [ ] **Low-confidence hold** — Cup your hand ambiguously; the bar should stall at
      partial fill and show a calibration hint ("Move hand to centre" etc.).
- [ ] **Voice on/off** — Click 🔊 Voice; committed letters should be spoken aloud.
      Click again; voice should stop immediately (`speechSynthesis.cancel()`).
- [ ] **Debug panel** — Press `?`; verify Stability row shows `heldMs / 750 ms`
      and Top-2 gap turns red when the margin is below threshold.

### Testing H vs nearby confusions

H is the most commonly confused letter in this dataset. Use the debug panel:

1. Press `?` to open the overlay.
2. Sign H (index + middle extended, pointing **right**):
   - **idxDX** (feat[18]) should be the dominant non-zero value.
   - **idxDY** (feat[17]) should be smaller in magnitude than **idxDX**.
   - RF label should show `H`; ⚠H indicator should **not** appear.
3. Sign U (index + middle extended, pointing **up**):
   - **idxDY** should dominate.
   - ⚠H indicator appears → model overrides to U or V based on vote count.
4. If the override misfires, check that the vote ratio `uV/hV` or `vV/hV`
   exceeds 0.55. Adjust the constant in `onHandResults` near `hV*0.55`.

---

## Quick smoke test — verify the full pipeline structure

This downloads 2 real clips per class, runs extraction, trains for 3 epochs,
and verifies the ONNX export. Takes ~5 minutes with a working internet connection.

```bash
pip install mediapipe opencv-python torch scikit-learn yt-dlp
# ffmpeg must be on PATH

python run_pipeline.py --smoke-test
```

Expected output at the end:
```
Step B  Prepare manifest CSV            OK
Step C  Download + trim clips           OK
Step D  Extract landmark sequences      OK
Step E  Train GRU classifier           OK
Step F  Export model to TF.js          OK   (or FAILED if onnx2tf not installed)

All smoke tests passed. Full pipeline is structurally sound.
```

---

## Full pipeline — step by step

### Prerequisites

```bash
pip install mediapipe opencv-python numpy torch scikit-learn matplotlib yt-dlp
pip install onnx onnx2tf tensorflowjs   # for Step F (model export)
# Install ffmpeg: apt install ffmpeg  OR  brew install ffmpeg
```

### Option A — Run everything automatically

```bash
python run_pipeline.py --workers 4 --min-clips 15
```

The orchestrator runs Steps B→F in order. If the clip threshold is not met
after Step C, it stops and prints a detailed report. Re-run after adjusting
`--min-clips` or after waiting for dead links to recover.

To resume from a specific step:
```bash
python run_pipeline.py --from-step D   # skip B+C if clips already downloaded
python run_pipeline.py --from-step E   # skip B+C+D if sequences already extracted
```

### Option B — Run each step manually

#### Step B — Build manifest

```bash
python data/prepare_manifest.py \
    --msasl-dir MS-ASL \
    --out data/processed/msasl_5word_manifest.csv
```

**Expected output:**
```
Resolved vocabulary:
  hello           → MS-ASL label   0  ('hello')
  how             → MS-ASL label  59  ('how')
  how_are_you     → MS-ASL label 274  ('how are you')
  you             → MS-ASL label  68  ('you')
  today           → MS-ASL label 131  ('today')
Manifest written → data/processed/msasl_5word_manifest.csv  (241 rows)
TRAIN  (142 clips total)
  hello            30
  how              32
  how_are_you      19
  you              35
  today            26
```

**Files created:**
```
data/processed/msasl_5word_manifest.csv
data/processed/label_to_id.json
data/processed/id_to_label.json
```

**Smoke test:**
```bash
python data/prepare_manifest.py --smoke-test 3
# → 45-row manifest (3 per class per split), takes <5s
```

---

#### Step C — Download clips

```bash
python data/download_clips.py \
    --manifest data/processed/msasl_5word_manifest.csv \
    --out-dir  data/raw/msasl_5word \
    --workers  4 \
    --min-clips 15
```

**Expected output (per clip):**
```
[1/241] OK    hello          ...youtube.com/watch?v=...
[2/241] FAIL  how            yt-dlp rc=1: Video unavailable
[3/241] SKIP  you            (already exists on disk)
...
── Download report ────────────────────────────────
Total:   241
OK:      ~150  (newly downloaded)
Skipped: 0    (already existed)
Failed:  ~91  (dead links / errors)
...
Threshold check PASSED — all train classes have >= 15 clips.
```

**Files created:**
```
data/raw/msasl_5word/train/{hello,how,how_are_you,you,today}/*.mp4
data/raw/msasl_5word/val/{...}/*.mp4
data/raw/msasl_5word/test/{...}/*.mp4
logs/download_report.json
```

If the threshold is NOT met, exit code is `2` and output shows:
```
── THRESHOLD NOT MET — PIPELINE STOPPED ─────────────
Minimum required usable clips per train class: 15
  how_are_you       8 clips  *** BELOW THRESHOLD
Options:
  1. Lower --min-clips (currently 15) to match available data.
  2. Re-run with --workers 8 to retry failed downloads.
  3. Check logs/download_report.json for failure details.
```

**Smoke test (2 clips per class):**
```bash
python data/download_clips.py --smoke-test
# Downloads ~10 clips total, skips threshold check.
# Takes ~3 min with decent internet.
```

---

#### Step D — Extract landmark sequences

```bash
python features/extract_sequences.py \
    --raw-dir data/raw/msasl_5word \
    --out-dir data/processed \
    --seq-len 30
```

**Expected output:**
```
Processing split: train
  train/hello           ~18 clips
  train/how             ~22 clips
  train/how_are_you     ~13 clips
  train/you             ~24 clips
  train/today           ~17 clips
  Saved X_train.npy (94, 30, 63)  y_train.npy (94,)
...
Elapsed: 312.4s
Stats → data/processed/extraction_stats.json
```

**Files created:**
```
data/processed/X_train.npy   shape (N_train, 30, 63) float32
data/processed/y_train.npy   shape (N_train,)         int32
data/processed/X_val.npy
data/processed/y_val.npy
data/processed/X_test.npy
data/processed/y_test.npy
data/processed/extraction_stats.json
```

**Smoke test (4 clips per class):**
```bash
python features/extract_sequences.py --smoke-test 4
```

---

#### Step E — Train GRU classifier

```bash
python train/train_model.py \
    --data-dir data/processed \
    --ckpt-dir train/checkpoints \
    --epochs   60 \
    --hidden   128
```

**Expected output:**
```
Train: 94  Val: 32  Test: 20  |  shape=(_, 30, 63)
Parameters: 123,013
Epoch   1/60  tr=1.6192/0.221  va=1.5788/0.344  *
Epoch   2/60  tr=1.4821/0.389  va=1.4392/0.437  *
...
Epoch  42/60  tr=0.3012/0.912  va=0.5541/0.813  (pat 10/10)
Early stopping at epoch 42

── Test results ─────────────────────────────────────
Accuracy: 0.750  |  F1-macro: 0.731
```

**Files created:**
```
train/checkpoints/best_model.pt
train/checkpoints/train_history.json
train/checkpoints/metrics.json
train/checkpoints/confusion_matrix.png
```

**Smoke test (3 epochs):**
```bash
python train/train_model.py --smoke-test
```

---

#### Step F — Export model to TF.js

```bash
pip install onnx onnx2tf tensorflowjs   # one-time

python inference/export_tfjs.py \
    --ckpt train/checkpoints/best_model.pt \
    --out  model/word_gru
```

**Expected output:**
```
1/3  PyTorch → ONNX
ONNX exported → /tmp/.../model.onnx
2/3  ONNX → TF SavedModel
3/3  SavedModel → TF.js
TF.js model → model/word_gru
```

**Files created:**
```
model/word_gru/model.json
model/word_gru/group1-shard1of1.bin
model/word_gru/classes.json
```

**Smoke test (ONNX only, no TF deps needed):**
```bash
python inference/export_tfjs.py --smoke-test
```

---

#### Run the web app

```bash
python -m http.server 8080
# Open http://localhost:8080
# Click "Switch to Word Mode"
```

The word badge shows the predicted sign; the phrase bar assembles the sentence.
The model loads automatically from `model/word_gru/model.json`.

---

## Project layout

```
Hand_Gesture/
├── index.html                 # Web app (letter mode + word mode)
├── check_connections.py       # Camera + library check
├── run_pipeline.py            # Pipeline orchestrator (Steps B→F)
│
├── data/
│   ├── prepare_manifest.py    # Step B: manifest CSV
│   ├── download_clips.py      # Step C: yt-dlp + ffmpeg download
│   ├── processed/             # manifest CSV, .npy arrays, label maps
│   └── raw/msasl_5word/       # downloaded video clips
│
├── features/
│   └── extract_sequences.py   # Step D: MediaPipe landmark extraction
│
├── train/
│   ├── train_model.py         # Step E: GRU training
│   └── checkpoints/           # best_model.pt, metrics, plots
│
├── inference/
│   ├── phrase_assembler.py    # Phrase decoder (also used by index.html JS)
│   └── export_tfjs.py         # Step F: PT → TF.js conversion
│
├── model/
│   └── word_gru/              # TF.js model (created by Step F)
│
├── MS-ASL/                    # MS-ASL metadata JSONs
└── logs/
    └── download_report.json   # Dead links and per-class download counts
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `yt-dlp not found` | `pip install yt-dlp` |
| `ffmpeg not found` | `apt install ffmpeg` or `brew install ffmpeg` |
| Step C exit code 2 (threshold) | Lower `--min-clips` to 10; see `logs/download_report.json` |
| `how_are_you` below threshold | Expected — only 19 clips in MS-ASL; use `--min-clips 10` |
| Low detection rate warnings | Normal for low-quality YouTube clips; < 30% triggers a warning |
| Word model not loading | Run Steps E + F; check browser console for CORS errors |
| `onnx2tf` install fails | Use `pip install onnx2tf==1.26.3` (pinned version) |
| Browser: no word predictions | Ensure you're serving via HTTP, not file://) |

## Adding new signs

1. Add the new gloss to `VOCAB` in `data/prepare_manifest.py`.
2. Re-run Steps B → F.
3. Update `WORD_CLASSES` in `index.html` to match the new class list.
4. Update `TARGET_PHRASES` in `inference/phrase_assembler.py`.

## Live recognition update

The camera workspace now includes explicit Start/Stop controls, a mirrored feed
with an aligned landmark overlay, a separate recognition panel, readable history,
voice feedback, optional diagnostics, and a responsive mobile layout. Backspace
clears the active history; `?` toggles diagnostics. Camera permission failures
show an inline explanation and a retry button. Camera tracks stop on page exit.

Recognition fixes:

- Only one camera stream and one MediaPipe request run at a time, including
  across stop/restart. Duplicate video frames are skipped and hidden tabs pause
  processing. Capture requests up to 30 fps at an ideal 960 × 540 resolution.
- Letters require a 70% rolling majority that agrees with the current confident
  observation for the full hold interval. Low confidence resets that interval.
  Holding a letter commits once; lower your hand for 400 ms to repeat it.
- Motion letters enter history once per detected motion event. J/Z remain
  heuristic and require validation with real signing footage.
- Word inference requires 30 collected frames, runs at most every 200 ms,
  and never overlaps. Mode changes, hand loss, history clearing, and camera
  stops invalidate pending results. All temporary tensors are disposed.
- Browser word resampling uses the same floor indices as Python extraction.
- Word loading uses `tf.loadGraphModel`, matching the SavedModel export, checks
  `classes.json` order, and applies softmax once to the exported logits.

The optional word model is **not included in this repository**. Export it using
the training pipeline and provide `model/word_gru/model.json`, all referenced
weight shards, and `classes.json`. Word mode supports only the five trained
classes; it is not a general ASL translator. Live accuracy and latency depend on
the trained model, signing style, lighting, and device, and have not been
benchmarked by this update.

Run the dependency-free regression suite with Node.js:

```bash
node --test tests/recognition.test.cjs
```

Before releasing, test with a real camera: Start/Stop/restart, denied permission,
portrait/mobile resize, single held letters, repeated letters after lowering the
hand, J/Z, and the exported five-class word model. Automated tests use synthetic
landmarks and mocked inference; they do not establish recognition accuracy.
