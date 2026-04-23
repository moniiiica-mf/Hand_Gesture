"""
Step D — Landmark sequence extraction.

Reads each clip from data/raw/msasl_5word/{split}/{label}/*.mp4,
runs MediaPipe Hands on every frame, normalises the 21-landmark hand
skeleton (wrist-centred, palm-width-scaled), resamples to a fixed
sequence length, and saves NumPy arrays for model training.

Outputs (in data/processed/):
    X_{split}.npy   — float32, shape (N, SEQ_LEN, FEAT_DIM)
    y_{split}.npy   — int32,   shape (N,)
    extraction_stats.json

Feature layout (FEAT_DIM = 63):
    Landmarks 0-20, each (x, y, z), wrist-centred and divided by
    palm width (lm5→lm17 distance).  Missing detections → zero vector.

Usage:
    python features/extract_sequences.py \
        --raw-dir data/raw/msasl_5word \
        --out-dir data/processed \
        [--seq-len 30] [--splits train val test]

    # Smoke test — process at most 4 clips per class per split
    python features/extract_sequences.py --smoke-test 4
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp_lib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FEAT_DIM    = 63       # 21 landmarks × 3
PALM_SCALE_A = 5       # index-finger MCP
PALM_SCALE_B = 17      # pinky MCP
MIN_PALM_W  = 1e-4


# ── Frame-level feature extraction ────────────────────────────────────────────

def _frame_features(hand_lm) -> np.ndarray:
    lm = hand_lm.landmark
    wrist = np.array([lm[0].x, lm[0].y, lm[0].z], dtype=np.float32)
    a = np.array([lm[PALM_SCALE_A].x, lm[PALM_SCALE_A].y, lm[PALM_SCALE_A].z], dtype=np.float32)
    b = np.array([lm[PALM_SCALE_B].x, lm[PALM_SCALE_B].y, lm[PALM_SCALE_B].z], dtype=np.float32)
    pw = float(np.linalg.norm(a - b))
    if pw < MIN_PALM_W:
        return np.zeros(FEAT_DIM, dtype=np.float32)
    pts = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
    return ((pts - wrist) / pw).flatten()


def _resample(frames: list, seq_len: int) -> np.ndarray:
    T = len(frames)
    out = np.zeros((seq_len, FEAT_DIM), dtype=np.float32)
    if T == 0:
        return out
    if T >= seq_len:
        idx = np.linspace(0, T - 1, seq_len, dtype=int)
        for i, j in enumerate(idx):
            out[i] = frames[j]
    else:
        for i, f in enumerate(frames):
            out[i] = f
    return out


# ── Per-clip extraction ────────────────────────────────────────────────────────

def extract_clip(video_path: Path, seq_len: int) -> tuple[Optional[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, {"error": "cannot_open", "frames": 0, "detected": 0}

    hands = mp_lib.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    raw: list[np.ndarray] = []
    total = detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            raw.append(_frame_features(res.multi_hand_landmarks[0]))
            detected += 1
        else:
            raw.append(np.zeros(FEAT_DIM, dtype=np.float32))

    cap.release()
    hands.close()

    det_rate = detected / total if total else 0.0
    seq = _resample(raw, seq_len)
    return seq, {"frames": total, "detected": detected,
                 "detection_rate": round(det_rate, 3), "error": None}


# ── Split processing ───────────────────────────────────────────────────────────

def process_split(
    raw_dir: Path,
    split: str,
    label_to_id: dict,
    seq_len: int,
    smoke_n: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    split_dir = raw_dir / split
    if not split_dir.exists():
        log.warning("Split directory not found: %s — skipping", split_dir)
        return (np.empty((0, seq_len, FEAT_DIM), dtype=np.float32),
                np.empty(0, dtype=np.int32), {})

    X_list, y_list = [], []
    per_label: dict = {}
    total_clips = ok_clips = zero_det = 0

    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label not in label_to_id:
            log.debug("Unknown label dir '%s' — skipping", label)
            continue
        class_id = label_to_id[label]
        clips = sorted(label_dir.glob("*.mp4"))
        if smoke_n:
            clips = clips[:smoke_n]
        log.info("  %s/%-15s  %d clips", split, label, len(clips))

        ok = fail = 0
        det_rates: list[float] = []

        for clip in clips:
            total_clips += 1
            seq, stats = extract_clip(clip, seq_len)
            if seq is None or stats["frames"] == 0:
                fail += 1
                continue
            X_list.append(seq)
            y_list.append(class_id)
            ok += 1
            ok_clips += 1
            det_rates.append(stats["detection_rate"])
            if stats["detection_rate"] == 0.0:
                zero_det += 1

        mean_det = float(np.mean(det_rates)) if det_rates else 0.0
        per_label[label] = {"ok": ok, "fail": fail,
                            "mean_detection_rate": round(mean_det, 3)}

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, FEAT_DIM), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    stats_out = {
        "total_clips": total_clips, "ok_clips": ok_clips,
        "failed_clips": total_clips - ok_clips,
        "zero_detection_clips": zero_det,
        "per_label": per_label,
    }
    return X, y, stats_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand sequences")
    parser.add_argument("--raw-dir",  default="data/raw/msasl_5word")
    parser.add_argument("--out-dir",  default="data/processed")
    parser.add_argument("--seq-len",  type=int, default=30)
    parser.add_argument("--splits",   nargs="+", default=["train", "val", "test"])
    parser.add_argument("--label-map", default="data/processed/label_to_id.json")
    parser.add_argument("--smoke-test", type=int, default=0, metavar="N",
                        help="Process at most N clips per class per split")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map_path = Path(args.label_map)
    if not label_map_path.exists():
        log.error("label_to_id.json not found — run prepare_manifest.py first")
        raise SystemExit(1)
    label_to_id: dict = json.loads(label_map_path.read_text())
    id_to_label: dict = {str(v): k for k, v in label_to_id.items()}
    num_classes = len(label_to_id)

    if args.smoke_test:
        log.info("SMOKE TEST: processing at most %d clips per class per split", args.smoke_test)
    log.info("Classes: %s  |  SEQ_LEN=%d  FEAT_DIM=%d", list(label_to_id), args.seq_len, FEAT_DIM)

    all_stats: dict = {"seq_len": args.seq_len, "feat_dim": FEAT_DIM, "splits": {}}
    t0 = time.time()

    for split in args.splits:
        log.info("\nProcessing split: %s", split)
        X, y, ss = process_split(raw_dir, split, label_to_id, args.seq_len, args.smoke_test)

        np.save(out_dir / f"X_{split}.npy", X)
        np.save(out_dir / f"y_{split}.npy", y)
        log.info("  Saved X_%s.npy %s  y_%s.npy %s", split, X.shape, split, y.shape)

        unique, counts = np.unique(y, return_counts=True)
        ss["class_counts"] = {id_to_label.get(str(int(u)), str(u)): int(c)
                              for u, c in zip(unique, counts)}
        all_stats["splits"][split] = ss

        # Warn about low detection rates
        for label, ls in ss.get("per_label", {}).items():
            if ls["mean_detection_rate"] < 0.3 and ls["ok"] > 0:
                log.warning("  LOW detection rate for %s/%s: %.0f%%",
                            split, label, ls["mean_detection_rate"] * 100)

    elapsed = round(time.time() - t0, 1)
    all_stats["elapsed_seconds"] = elapsed
    (out_dir / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2))
    (out_dir / "id_to_label.json").write_text(json.dumps(id_to_label, indent=2))

    stats_path = out_dir / "extraction_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2))

    log.info("\n── Extraction summary ─────────────────────────────")
    log.info("Elapsed: %.1fs", elapsed)
    for split, ss in all_stats["splits"].items():
        log.info("%s: %d ok / %d fail  (zero-det clips: %d)",
                 split, ss["ok_clips"], ss["failed_clips"], ss["zero_detection_clips"])
    log.info("Stats → %s", stats_path)

    if args.smoke_test:
        log.info("\nSMOKE TEST PASSED — MediaPipe extraction pipeline is functional.")


if __name__ == "__main__":
    main()
