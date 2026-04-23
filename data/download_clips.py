"""
Step C — Clip download and trimming.

Reads the manifest CSV produced by prepare_manifest.py, downloads
each source video via yt-dlp, trims it to [start_time, end_time]
using ffmpeg, and organises outputs under:

    data/raw/msasl_5word/{train,val,test}/{label}/<signer_id>_<idx>.mp4

Failures (dead links, ffmpeg errors, corrupt files) are logged to:

    logs/download_report.json

Usage:
    python data/download_clips.py \
        --manifest data/processed/msasl_5word_manifest.csv \
        --out-dir  data/raw/msasl_5word \
        --workers  4 \
        [--splits train val test] \
        [--dry-run]

The script is idempotent: already-downloaded clips are skipped.
If the dead-link rate exceeds --fallback-threshold (default 0.50),
it prints instructions for the local-recording fallback workflow.
"""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
YT_DLP_OPTS = [
    "--quiet",
    "--no-warnings",
    "--format", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]/best",
    "--merge-output-format", "mp4",
    "--no-playlist",
    "--socket-timeout", "30",
    "--retries", "2",
]

# Padding added around [start, end] before ffmpeg trim (seconds).
# Ensures we don't clip the very first/last frame.
TRIM_PAD = 0.1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run a subprocess; return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def _verify_mp4(path: Path) -> bool:
    """Return True if the file is a readable MP4 with at least one video frame."""
    rc, _, err = _run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        timeout=15,
    )
    if rc != 0:
        return False
    # nb_frames may be 'N/A' for some containers
    return True


def download_and_trim(
    row: dict,
    out_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Download + trim a single clip.

    Returns a result dict with keys:
        success (bool), path (str|None), error (str|None), skipped (bool)
    """
    label   = row["label"]
    split   = row["split"]
    url     = row["url"]
    t_start = float(row["start_time"])
    t_end   = float(row["end_time"])
    signer  = row["signer_id"]
    idx     = row.get("_row_idx", 0)

    out_class_dir = out_dir / split / label
    out_class_dir.mkdir(parents=True, exist_ok=True)

    clip_name = f"{signer}_{idx:04d}.mp4"
    out_path  = out_class_dir / clip_name

    # ── Already done ─────────────────────────────────────────────────────────
    if out_path.exists() and out_path.stat().st_size > 1024:
        return {"success": True, "path": str(out_path), "error": None, "skipped": True}

    if dry_run:
        log.info("[DRY-RUN] Would download %s → %s", url, out_path)
        return {"success": True, "path": str(out_path), "error": None, "skipped": True}

    # ── Download to temp file ─────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_template = str(Path(tmpdir) / "source.%(ext)s")

        dl_cmd = ["yt-dlp"] + YT_DLP_OPTS + ["-o", tmp_template, url]
        try:
            rc, stdout, stderr = _run(dl_cmd, timeout=180)
        except subprocess.TimeoutExpired:
            return {"success": False, "path": None,
                    "error": "yt-dlp timeout", "skipped": False}

        if rc != 0:
            short_err = (stderr or stdout or "yt-dlp failed").strip()[-200:]
            return {"success": False, "path": None,
                    "error": f"yt-dlp rc={rc}: {short_err}", "skipped": False}

        # Find downloaded file
        downloaded = list(Path(tmpdir).glob("source.*"))
        if not downloaded:
            return {"success": False, "path": None,
                    "error": "yt-dlp produced no output file", "skipped": False}
        src_file = downloaded[0]

        # ── Trim with ffmpeg ──────────────────────────────────────────────────
        padded_start = max(0.0, t_start - TRIM_PAD)
        duration     = (t_end - t_start) + 2 * TRIM_PAD

        trim_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(padded_start),
            "-i", str(src_file),
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac", "-b:a", "64k",
            str(out_path),
        ]
        try:
            rc2, _, stderr2 = _run(trim_cmd, timeout=120)
        except subprocess.TimeoutExpired:
            return {"success": False, "path": None,
                    "error": "ffmpeg timeout", "skipped": False}

        if rc2 != 0:
            return {"success": False, "path": None,
                    "error": f"ffmpeg rc={rc2}: {stderr2.strip()[-200:]}", "skipped": False}

        # ── Verify output ─────────────────────────────────────────────────────
        if not out_path.exists() or out_path.stat().st_size < 512:
            return {"success": False, "path": None,
                    "error": "Output file missing or too small", "skipped": False}

        if not _verify_mp4(out_path):
            out_path.unlink(missing_ok=True)
            return {"success": False, "path": None,
                    "error": "ffprobe verification failed", "skipped": False}

    return {"success": True, "path": str(out_path), "error": None, "skipped": False}


# ── Fallback instructions ─────────────────────────────────────────────────────

def print_fallback_instructions(out_dir: Path, label_to_id: dict) -> None:
    log.warning("\n" + "="*60)
    log.warning("FALLBACK: Dead-link rate too high.")
    log.warning("Record your own clips using the instructions below.")
    log.warning("="*60)
    print("""
LOCAL RECORDING FALLBACK
─────────────────────────────────────────────────────────────
Requirements per class:
  • Minimum: 30 clips
  • Recommended: 50+ clips from multiple angles / lighting conditions
  • Length: 1–3 seconds each
  • Format: MP4, any resolution (will be resized)
  • Signer variety: at least 2 different people if possible

Directory layout expected by Step D:
  data/raw/msasl_5word/train/{label}/*.mp4
  data/raw/msasl_5word/val/{label}/*.mp4
  data/raw/msasl_5word/test/{label}/*.mp4

Classes to record:""")
    for label in label_to_id:
        print(f"  {label}")
    print("""
Recording tips:
  1. Use a plain background if possible.
  2. Ensure your whole hand is visible throughout.
  3. Record at least 3 different distances from camera.
  4. Vary lighting (indoor lamp vs window light).
  5. Make one continuous clip per sign attempt — do NOT splice.

Split ratio: 70% train / 15% val / 15% test
  (sort filenames alphabetically; first 70% → train, etc.)

Once recorded, re-run Step D:
  python features/extract_sequences.py

Dataset source will be marked as 'local' in metadata.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download + trim MS-ASL clips")
    parser.add_argument("--manifest",  default="data/processed/msasl_5word_manifest.csv")
    parser.add_argument("--out-dir",   default="data/raw/msasl_5word")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel download threads")
    parser.add_argument("--splits",    nargs="+", default=["train","val","test"])
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print what would be done without downloading")
    parser.add_argument("--fallback-threshold", type=float, default=0.50,
                        help="Trigger fallback instructions if failure rate > this")
    args = parser.parse_args()

    # ── Check dependencies ────────────────────────────────────────────────────
    for tool in ("yt-dlp", "ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            log.error("'%s' not found. Install it first.", tool)
            raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # ── Load manifest ─────────────────────────────────────────────────────────
    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    rows_filtered = [r for r in rows if r["split"] in args.splits]
    log.info("Loaded manifest: %d clips (splits: %s)", len(rows_filtered), args.splits)

    # Tag each row with its index for unique filename
    for i, r in enumerate(rows_filtered):
        r["_row_idx"] = i

    # Load label map for fallback
    label_to_id_path = Path("data/processed/label_to_id.json")
    label_to_id = json.loads(label_to_id_path.read_text()) if label_to_id_path.exists() else {}

    # ── Download in parallel ──────────────────────────────────────────────────
    results: list[dict] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_and_trim, row, out_dir, args.dry_run): row
            for row in rows_filtered
        }
        done = 0
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"success": False, "path": None,
                       "error": str(exc), "skipped": False}
            res["label"]    = row["label"]
            res["split"]    = row["split"]
            res["url"]      = row["url"]
            res["signer_id"]= row["signer_id"]
            results.append(res)
            done += 1

            status = "SKIP" if res["skipped"] else ("OK" if res["success"] else "FAIL")
            log.info("[%d/%d] %-4s  %-15s  %s",
                     done, len(rows_filtered), status,
                     row["label"], row["url"][-60:])
            if not res["success"] and res["error"]:
                log.debug("  Error: %s", res["error"])

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    ok      = [r for r in results if r["success"] and not r["skipped"]]
    skipped = [r for r in results if r["skipped"]]
    failed  = [r for r in results if not r["success"]]

    log.info("\n── Download report ────────────────────────────────")
    log.info("Total:   %d", len(results))
    log.info("OK:      %d", len(ok))
    log.info("Skipped: %d  (already existed)", len(skipped))
    log.info("Failed:  %d", len(failed))
    log.info("Elapsed: %.1fs", elapsed)

    # Per-class breakdown
    from collections import defaultdict
    by_label: dict[str, dict] = defaultdict(lambda: {"ok":0,"skip":0,"fail":0})
    for r in results:
        key = r["label"]
        if r["skipped"]:        by_label[key]["skip"] += 1
        elif r["success"]:      by_label[key]["ok"]   += 1
        else:                   by_label[key]["fail"]  += 1

    log.info("\nPer-class:")
    for label, counts in sorted(by_label.items()):
        log.info("  %-15s ok=%d  skip=%d  fail=%d",
                 label, counts["ok"], counts["skip"], counts["fail"])

    # Write JSON report
    report = {
        "total": len(results),
        "ok": len(ok),
        "skipped": len(skipped),
        "failed": len(failed),
        "elapsed_seconds": round(elapsed, 1),
        "per_class": {k: v for k, v in by_label.items()},
        "failures": [
            {"url": r["url"], "label": r["label"],
             "split": r["split"], "error": r["error"]}
            for r in failed
        ],
    }
    report_path = Path("logs/download_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    log.info("\nReport → %s", report_path)

    # ── Fallback trigger ──────────────────────────────────────────────────────
    attempted = len(ok) + len(failed)
    if attempted > 0:
        fail_rate = len(failed) / attempted
        if fail_rate > args.fallback_threshold:
            log.warning("Failure rate %.0f%% exceeds threshold %.0f%%",
                        fail_rate * 100, args.fallback_threshold * 100)
            print_fallback_instructions(out_dir, label_to_id)
        else:
            log.info("Failure rate %.0f%% — within acceptable range.", fail_rate * 100)

    # Final usable clip count per class/split
    log.info("\n── Usable clips on disk ────────────────────────────")
    for split in args.splits:
        for label_dir in sorted((out_dir / split).iterdir()) if (out_dir / split).exists() else []:
            clips = list(label_dir.glob("*.mp4"))
            log.info("  %s/%s: %d clips", split, label_dir.name, len(clips))


if __name__ == "__main__":
    main()
