"""
Step D — Landmark sequence extraction.

Reads each clip from data/raw/msasl_5word/{split}/{label}/*.mp4,
runs MediaPipe Hands on every frame, normalises the 21-landmark hand
skeleton (wrist-centred, palm-width-scaled), resamples the sequence
to a fixed length, and saves NumPy arrays ready for model training.

Outputs (in data/processed/):
    X_{split}.npy   — float32, shape (N, SEQ_LEN, FEAT_DIM)
    y_{split}.npy   — int32,   shape (N,)
    label_to_id.json  (copied/verified from manifest step)
    id_to_label.json
    extraction_stats.json

Usage:
    python features/extract_sequences.py \
        --raw-dir   data/raw/msasl_5word \
        --out-dir   data/processed \
        --seq-len   30 \
        --workers   4 \
        [--splits train val test] \
        [--max-hands 1]

Feature layout (FEAT_DIM = 63):
    landmarks 0-20, each with (x, y, z), all wrist-centred and
    divided by palm width (lm5→lm17 distance).  Missing detections
    → zero vector for that frame.
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
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

# ── Constants ──────────────────────────────────────────────────────────────────
FEAT_DIM = 63          # 21 landmarks × 3 (x, y, z)
# Landmarks used to compute palm width for scale normalisation
PALM_SCALE_A = 5       # index-finger MCP
PALM_SCALE_B = 17      # pinky MCP
MIN_PALM_WIDTH = 1e-4  # guard against near-zero division


# ── Feature helpers ────────────────────────────────────────────────────────────

def _extract_frame_features(hand_landmarks) -> np.ndarray:
    """
    Convert one MediaPipe NormalizedLandmarkList to a FEAT_DIM float32 vector.
    Landmarks are wrist-centred and palm-width-scaled.
    Returns a zero vector if the hand is degenerate (palm width too small).
    """
    lm = hand_landmarks.landmark
    wrist = np.array([lm[0].x, lm[0].y, lm[0].z], dtype=np.float32)
    a = np.array([lm[PALM_SCALE_A].x, lm[PALM_SCALE_A].y, lm[PALM_SCALE_A].z], dtype=np.float32)
    b = np.array([lm[PALM_SCALE_B].x, lm[PALM_SCALE_B].y, lm[PALM_SCALE_B].z], dtype=np.float32)
    palm_width = float(np.linalg.norm(a - b))

    if palm_width < MIN_PALM_WIDTH:
        return np.zeros(FEAT_DIM, dtype=np.float32)

    pts = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)  # (21, 3)
    pts = (pts - wrist) / palm_width
    return pts.flatten()  # (63,)


def _resample_sequence(frames: list[np.ndarray], seq_len: int) -> np.ndarray:
    """
    Given a list of FEAT_DIM vectors (one per frame with a detection),
    return a (seq_len, FEAT_DIM) array by:
      - Uniform sampling if len > seq_len
      - Zero-padding at the end if len < seq_len
      - Direct copy if len == seq_len
    """
    T = len(frames)
    out = np.zeros((seq_len, FEAT_DIM), dtype=np.float32)

    if T == 0:
        return out

    if T >= seq_len:
        # Uniform subsample: pick seq_len evenly-spaced indices
        indices = np.linspace(0, T - 1, seq_len, dtype=int)
        for i, idx in enumerate(indices):
            out[i] = frames[idx]
    else:
        # Copy all frames; remainder stays zero (end-padding)
        for i, f in enumerate(frames):
            out[i] = f

    return out


# ── Per-clip extraction ────────────────────────────────────────────────────────

def extract_clip(
    video_path: Path,
    seq_len: int,
    max_hands: int = 1,
) -> tuple[Optional[np.ndarray], dict]:
    """
    Process one video file. Returns (seq_array, stats_dict).
    seq_array is (seq_len, FEAT_DIM) float32, or None on hard failure.
    stats_dict contains per-clip diagnostics.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, {"error": "cannot_open", "frames": 0, "detected": 0}

    hands = mp_lib.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    raw_frames: list[np.ndarray] = []   # all frame feature vectors (zeros for misses)
    detected_count = 0
    total_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            feat = _extract_frame_features(result.multi_hand_landmarks[0])
            detected_count += 1
        else:
            feat = np.zeros(FEAT_DIM, dtype=np.float32)

        raw_frames.append(feat)

    cap.release()
    hands.close()

    detection_rate = detected_count / total_count if total_count > 0 else 0.0
    seq = _resample_sequence(raw_frames, seq_len)
    stats = {
        "frames": total_count,
        "detected": detected_count,
        "detection_rate": round(detection_rate, 3),
        "error": None,
    }
    return seq, stats


# ── Split processing ───────────────────────────────────────────────────────────

def process_split(
    raw_dir: Path,
    split: str,
    label_to_id: dict,
    seq_len: int,
    max_hands: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Walk raw_dir/split/{label}/*.mp4, extract features, return arrays + stats.
    """
    split_dir = raw_dir / split
    if not split_dir.exists():
        log.warning("Split directory not found: %s — skipping", split_dir)
        return np.empty((0, seq_len, FEAT_DIM), dtype=np.float32), np.empty(0, dtype=np.int32), {}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    per_label_stats: dict = {}
    total_clips = 0
    ok_clips = 0
    zero_detection_clips = 0

    label_dirs = sorted(split_dir.iterdir())
    for label_dir in label_dirs:
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label not in label_to_id:
            log.debug("Unknown label directory '%s' — skipping", label)
            continue
        class_id = label_to_id[label]

        clips = sorted(label_dir.glob("*.mp4"))
        log.info("  %s/%s: %d clips", split, label, len(clips))

        label_ok = 0
        label_fail = 0
        label_det_rates: list[float] = []

        for clip in clips:
            total_clips += 1
            seq, stats = extract_clip(clip, seq_len, max_hands)

            if seq is None or stats.get("error") == "cannot_open":
                log.debug("    FAIL  %s  (%s)", clip.name, stats.get("error"))
                label_fail += 1
                continue

            if stats["frames"] == 0:
                log.debug("    EMPTY %s", clip.name)
                label_fail += 1
                continue

            X_list.append(seq)
            y_list.append(class_id)
            label_ok += 1
            ok_clips += 1
            label_det_rates.append(stats["detection_rate"])

            if stats["detection_rate"] == 0.0:
                zero_detection_clips += 1

        mean_det = float(np.mean(label_det_rates)) if label_det_rates else 0.0
        per_label_stats[label] = {
            "ok": label_ok,
            "fail": label_fail,
            "mean_detection_rate": round(mean_det, 3),
        }

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, seq_len, FEAT_DIM), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    split_stats = {
        "total_clips": total_clips,
        "ok_clips": ok_clips,
        "failed_clips": total_clips - ok_clips,
        "zero_detection_clips": zero_detection_clips,
        "per_label": per_label_stats,
    }
    return X, y, split_stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand sequences from clips")
    parser.add_argument("--raw-dir",   default="data/raw/msasl_5word",
                        help="Root of downloaded clips")
    parser.add_argument("--out-dir",   default="data/processed",
                        help="Where to write .npy arrays and stats")
    parser.add_argument("--seq-len",   type=int, default=30,
                        help="Fixed sequence length (frames)")
    parser.add_argument("--splits",    nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-hands", type=int, default=1,
                        help="Max hands per frame (1 recommended for speed)")
    parser.add_argument("--label-map", default="data/processed/label_to_id.json",
                        help="label_to_id.json from prepare_manifest step")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load label map
    label_map_path = Path(args.label_map)
    if not label_map_path.exists():
        log.error("label_to_id.json not found at %s — run prepare_manifest.py first", label_map_path)
        raise SystemExit(1)
    label_to_id: dict = json.loads(label_map_path.read_text())
    id_to_label: dict = {str(v): k for k, v in label_to_id.items()}
    log.info("Classes: %s", list(label_to_id.keys()))
    log.info("SEQ_LEN=%d  FEAT_DIM=%d", args.seq_len, FEAT_DIM)

    all_stats: dict = {"seq_len": args.seq_len, "feat_dim": FEAT_DIM, "splits": {}}
    t0 = time.time()

    for split in args.splits:
        log.info("\nProcessing split: %s", split)
        X, y, split_stats = process_split(
            raw_dir, split, label_to_id, args.seq_len, args.max_hands
        )

        np.save(out_dir / f"X_{split}.npy", X)
        np.save(out_dir / f"y_{split}.npy", y)

        log.info("  Saved X_%s.npy  shape=%s  y_%s.npy  shape=%s",
                 split, X.shape, split, y.shape)

        # Per-class counts in saved arrays
        unique, counts = np.unique(y, return_counts=True)
        class_counts = {id_to_label.get(str(int(u)), str(u)): int(c)
                        for u, c in zip(unique, counts)}
        split_stats["class_counts"] = class_counts
        all_stats["splits"][split] = split_stats

    elapsed = round(time.time() - t0, 1)
    all_stats["elapsed_seconds"] = elapsed

    # Write label maps (ensure they exist in out_dir)
    (out_dir / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2))
    (out_dir / "id_to_label.json").write_text(json.dumps(id_to_label, indent=2))

    stats_path = out_dir / "extraction_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n── Extraction summary ──────────────────────────────")
    log.info("Elapsed: %.1fs", elapsed)
    for split, ss in all_stats["splits"].items():
        log.info("%s: %d ok / %d fail  |  zero-detection clips: %d",
                 split, ss["ok_clips"], ss["failed_clips"], ss["zero_detection_clips"])
        for label, ls in ss.get("per_label", {}).items():
            log.info("    %-15s ok=%d  fail=%d  mean_det=%.1f%%",
                     label, ls["ok"], ls["fail"], ls["mean_detection_rate"] * 100)
    log.info("Stats → %s", stats_path)

    # Sanity: warn if any split has < 5 samples per class
    for split, ss in all_stats["splits"].items():
        for label, cnt in ss.get("class_counts", {}).items():
            if cnt < 5:
                log.warning("LOW SAMPLE COUNT: %s/%s has only %d samples", split, label, cnt)


if __name__ == "__main__":
    main()
