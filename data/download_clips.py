"""
Step C — Clip download and trimming.

Reads the manifest CSV produced by prepare_manifest.py, downloads each
source video via yt-dlp, trims to [start_time, end_time] with ffmpeg, and
organises outputs under:

    data/raw/msasl_5word/{train,val,test}/{label}/<signer_id>_<idx>.mp4

Dead links and ffmpeg failures are logged to:

    logs/download_report.json

After all downloads, the script counts usable clips per class in the TRAIN
split.  If any class falls below --min-clips (default 15), it writes a
detailed failure report and exits with code 2 so the pipeline runner can
stop cleanly.

Usage:
    python data/download_clips.py \
        --manifest data/processed/msasl_5word_manifest.csv \
        --out-dir  data/raw/msasl_5word \
        --workers  4 \
        [--splits train val test] \
        [--min-clips 15] \
        [--dry-run]

    # Smoke test — download only 2 clips per class, no threshold check
    python data/download_clips.py --smoke-test

Exit codes:
    0  All train classes meet the minimum clip threshold.
    2  One or more train classes are below the threshold — pipeline stops.
    1  Hard error (missing dependency, bad manifest, etc.)
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

YT_DLP_OPTS = [
    "--quiet", "--no-warnings",
    "--format", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]/best",
    "--merge-output-format", "mp4",
    "--no-playlist",
    "--socket-timeout", "30",
    "--retries", "2",
]
TRIM_PAD = 0.1   # seconds of padding either side of [start, end]
SMOKE_PER_CLASS = 2   # clips per class when --smoke-test is active


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def _verify_mp4(path: Path) -> bool:
    rc, _, _ = _run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        timeout=15,
    )
    return rc == 0


def download_and_trim(row: dict, out_dir: Path, dry_run: bool = False) -> dict:
    """Download + trim one clip. Returns result dict."""
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

    if out_path.exists() and out_path.stat().st_size > 1024:
        return {"success": True, "path": str(out_path), "error": None, "skipped": True}

    if dry_run:
        log.info("[DRY-RUN] Would download %s → %s", url, out_path)
        return {"success": True, "path": str(out_path), "error": None, "skipped": True}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_template = str(Path(tmpdir) / "source.%(ext)s")
        dl_cmd = ["yt-dlp"] + YT_DLP_OPTS + ["-o", tmp_template, url]
        try:
            rc, stdout, stderr = _run(dl_cmd, timeout=180)
        except subprocess.TimeoutExpired:
            return {"success": False, "path": None, "error": "yt-dlp timeout", "skipped": False}

        if rc != 0:
            short_err = (stderr or stdout or "yt-dlp failed").strip()[-200:]
            return {"success": False, "path": None,
                    "error": f"yt-dlp rc={rc}: {short_err}", "skipped": False}

        downloaded = list(Path(tmpdir).glob("source.*"))
        if not downloaded:
            return {"success": False, "path": None,
                    "error": "yt-dlp produced no output file", "skipped": False}

        src_file = downloaded[0]
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
            return {"success": False, "path": None, "error": "ffmpeg timeout", "skipped": False}

        if rc2 != 0:
            return {"success": False, "path": None,
                    "error": f"ffmpeg rc={rc2}: {stderr2.strip()[-200:]}", "skipped": False}

        if not out_path.exists() or out_path.stat().st_size < 512:
            return {"success": False, "path": None,
                    "error": "Output file missing or too small", "skipped": False}

        if not _verify_mp4(out_path):
            out_path.unlink(missing_ok=True)
            return {"success": False, "path": None,
                    "error": "ffprobe verification failed", "skipped": False}

    return {"success": True, "path": str(out_path), "error": None, "skipped": False}


# ── Threshold check ────────────────────────────────────────────────────────────

def check_threshold(out_dir: Path, min_clips: int) -> tuple[bool, dict[str, int]]:
    """
    Count usable clips per class in the train split.
    Returns (all_meet_threshold, {label: count}).
    """
    train_dir = out_dir / "train"
    counts: dict[str, int] = {}
    if train_dir.exists():
        for label_dir in sorted(train_dir.iterdir()):
            if label_dir.is_dir():
                counts[label_dir.name] = len(list(label_dir.glob("*.mp4")))
    all_ok = all(v >= min_clips for v in counts.values()) if counts else False
    return all_ok, counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download + trim MS-ASL clips")
    parser.add_argument("--manifest",  default="data/processed/msasl_5word_manifest.csv")
    parser.add_argument("--out-dir",   default="data/raw/msasl_5word")
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--splits",    nargs="+", default=["train", "val", "test"])
    parser.add_argument("--min-clips", type=int, default=15,
                        help="Minimum usable clips per class in train split")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"Download only {SMOKE_PER_CLASS} clips per class; skip threshold check")
    args = parser.parse_args()

    for tool in ("yt-dlp", "ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            log.error("'%s' not found. Install it and retry.", tool)
            raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    rows_filtered = [r for r in rows if r["split"] in args.splits]

    # Smoke-test: keep only SMOKE_PER_CLASS per class per split
    if args.smoke_test:
        log.info("SMOKE TEST: keeping %d clips per class per split", SMOKE_PER_CLASS)
        counts: dict[str, int] = defaultdict(int)
        kept = []
        for r in rows_filtered:
            key = f"{r['split']}/{r['label']}"
            if counts[key] < SMOKE_PER_CLASS:
                kept.append(r)
                counts[key] += 1
        rows_filtered = kept

    log.info("Loaded manifest: %d clips (splits: %s)", len(rows_filtered), args.splits)
    for i, r in enumerate(rows_filtered):
        r["_row_idx"] = i

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
                res = {"success": False, "path": None, "error": str(exc), "skipped": False}
            res["label"]     = row["label"]
            res["split"]     = row["split"]
            res["url"]       = row["url"]
            res["signer_id"] = row["signer_id"]
            results.append(res)
            done += 1

            status = "SKIP" if res["skipped"] else ("OK" if res["success"] else "FAIL")
            log.info("[%d/%d] %-4s  %-15s  %s",
                     done, len(rows_filtered), status,
                     row["label"], row["url"][-60:])
            if not res["success"] and res["error"]:
                log.debug("  Error: %s", res["error"])

    elapsed = time.time() - t0

    ok      = [r for r in results if r["success"] and not r["skipped"]]
    skipped = [r for r in results if r["skipped"]]
    failed  = [r for r in results if not r["success"]]

    log.info("\n── Download report ────────────────────────────────")
    log.info("Total:   %d", len(results))
    log.info("OK:      %d  (newly downloaded)", len(ok))
    log.info("Skipped: %d  (already existed)", len(skipped))
    log.info("Failed:  %d  (dead links / errors)", len(failed))
    log.info("Elapsed: %.1fs", elapsed)

    by_label: dict[str, dict] = defaultdict(lambda: {"ok": 0, "skip": 0, "fail": 0})
    for r in results:
        key = r["label"]
        if r["skipped"]:        by_label[key]["skip"] += 1
        elif r["success"]:      by_label[key]["ok"]   += 1
        else:                   by_label[key]["fail"]  += 1

    log.info("\nPer-class:")
    for label, counts in sorted(by_label.items()):
        log.info("  %-15s ok=%d  skip=%d  fail=%d",
                 label, counts["ok"], counts["skip"], counts["fail"])

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
    log.info("\nFull report → %s", report_path)

    # ── Usable clips on disk ──────────────────────────────────────────────────
    log.info("\n── Usable clips on disk ─────────────────────────────")
    for split in args.splits:
        split_dir = out_dir / split
        if not split_dir.exists():
            continue
        for label_dir in sorted(split_dir.iterdir()):
            clips = list(label_dir.glob("*.mp4"))
            log.info("  %s/%-15s  %d clips", split, label_dir.name, len(clips))

    # ── Threshold check (train split only) ────────────────────────────────────
    if args.smoke_test:
        log.info("\nSMOKE TEST: skipping threshold check.")
        log.info("Smoke test PASSED — yt-dlp + ffmpeg pipeline is functional.")
        raise SystemExit(0)

    all_ok, train_counts = check_threshold(out_dir, args.min_clips)

    if not all_ok:
        log.error("\n── THRESHOLD NOT MET — PIPELINE STOPPED ─────────────")
        log.error("Minimum required usable clips per train class: %d", args.min_clips)
        log.error("")
        for label, cnt in sorted(train_counts.items()):
            flag = " *** BELOW THRESHOLD" if cnt < args.min_clips else " OK"
            log.error("  %-15s  %2d clips%s", label, cnt, flag)
        log.error("")
        log.error("Options:")
        log.error("  1. Lower --min-clips (currently %d) to match available data.", args.min_clips)
        log.error("  2. Re-run with --workers 8 to retry failed downloads.")
        log.error("  3. Check logs/download_report.json for failure details.")
        report["threshold_check"] = {"min_clips": args.min_clips,
                                     "train_counts": train_counts, "passed": False}
        report_path.write_text(json.dumps(report, indent=2))
        raise SystemExit(2)   # code 2 = threshold not met

    log.info("\nThreshold check PASSED — all train classes have >= %d clips.", args.min_clips)
    report["threshold_check"] = {"min_clips": args.min_clips,
                                 "train_counts": train_counts, "passed": True}
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
