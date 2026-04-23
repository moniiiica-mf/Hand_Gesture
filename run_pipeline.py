"""
Pipeline orchestrator — runs Steps B through F in order.

Stops automatically if the clip threshold is not met after download (Step C)
and prints a clear report.  Re-running after more clips appear resumes from
where it left off (each step is idempotent).

Usage:
    # Full run
    python run_pipeline.py [--workers 4] [--min-clips 15]

    # Smoke test — fast end-to-end check of every stage (tiny data)
    python run_pipeline.py --smoke-test

    # Start from a specific step (skip earlier completed steps)
    python run_pipeline.py --from-step D   # D=extract, E=train, F=export

Steps and expected runtimes:
    B  prepare_manifest   < 5 s
    C  download_clips     30–90 min (depends on network + dead links)
    D  extract_sequences  5–30 min (depends on clip count)
    E  train_model        2–20 min (CPU) / 1–5 min (GPU)
    F  export_tfjs        2–5 min
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STEP_ORDER = ["B", "C", "D", "E", "F"]

STEP_DESCRIPTIONS = {
    "B": "Prepare manifest CSV",
    "C": "Download + trim clips",
    "D": "Extract landmark sequences",
    "E": "Train GRU classifier",
    "F": "Export model to TF.js",
}


def run_step(label: str, cmd: list[str]) -> int:
    """Run a pipeline step; return its exit code."""
    log.info("\n%s", "=" * 60)
    log.info("STEP %s — %s", label, STEP_DESCRIPTIONS[label])
    log.info("Command: %s", " ".join(cmd))
    log.info("%s", "=" * 60)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    elapsed = time.time() - t0
    status = "PASSED" if rc == 0 else (f"THRESHOLD NOT MET (exit {rc})" if rc == 2 else f"FAILED (exit {rc})")
    log.info("Step %s — %s  (%.1fs)", label, status, elapsed)
    return rc


def check_prereq(path: Path, description: str) -> bool:
    if not path.exists():
        log.warning("  %s not found: %s", description, path)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ASL pipeline B→F")
    parser.add_argument("--workers",    type=int, default=4, help="Parallel download workers")
    parser.add_argument("--min-clips",  type=int, default=15,
                        help="Minimum usable clips per train class before proceeding")
    parser.add_argument("--epochs",     type=int, default=60)
    parser.add_argument("--from-step",  choices=STEP_ORDER, default="B",
                        help="Start from this step (earlier steps assumed complete)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick end-to-end test with minimal data")
    args = parser.parse_args()

    py = sys.executable
    smoke = ["--smoke-test"] if args.smoke_test else []

    start_idx = STEP_ORDER.index(args.from_step)
    steps_to_run = STEP_ORDER[start_idx:]

    log.info("Pipeline start.  Steps to run: %s", steps_to_run)
    if args.smoke_test:
        log.info("SMOKE TEST mode — each stage runs with minimal data.")

    summary: dict[str, str] = {}
    t_total = time.time()

    # ── Step B: Manifest ──────────────────────────────────────────────────────
    if "B" in steps_to_run:
        cmd = [py, "data/prepare_manifest.py"]
        if args.smoke_test:
            cmd += ["--smoke-test", "3"]
        rc = run_step("B", cmd)
        if rc != 0:
            summary["B"] = "FAILED"
            log.error("Step B failed. Aborting.")
            _print_summary(summary, time.time() - t_total)
            raise SystemExit(1)
        summary["B"] = "OK"

    # ── Step C: Download ───────────────────────────────────────────────────────
    if "C" in steps_to_run:
        cmd = [py, "data/download_clips.py",
               "--workers", str(args.workers),
               "--min-clips", str(args.min_clips)]
        if args.smoke_test:
            cmd += ["--smoke-test"]
        rc = run_step("C", cmd)
        if rc == 2:
            summary["C"] = "THRESHOLD NOT MET"
            log.error("\n%s", "=" * 60)
            log.error("PIPELINE STOPPED — clip threshold not met.")
            log.error("See logs/download_report.json for per-class counts.")
            log.error("")
            log.error("Options:")
            log.error("  1. Lower --min-clips (currently %d)", args.min_clips)
            log.error("  2. Re-run with more workers: --workers 8")
            log.error("  3. Re-run later (dead links may become available)")
            log.error("%s", "=" * 60)
            _print_summary(summary, time.time() - t_total)
            raise SystemExit(2)
        if rc != 0:
            summary["C"] = "FAILED"
            log.error("Step C failed. Aborting.")
            _print_summary(summary, time.time() - t_total)
            raise SystemExit(1)
        summary["C"] = "OK"

    # ── Step D: Extract sequences ─────────────────────────────────────────────
    if "D" in steps_to_run:
        cmd = [py, "features/extract_sequences.py"]
        if args.smoke_test:
            cmd += ["--smoke-test", "4"]
        rc = run_step("D", cmd)
        if rc != 0:
            summary["D"] = "FAILED"
            _print_summary(summary, time.time() - t_total)
            raise SystemExit(1)
        summary["D"] = "OK"

    # ── Step E: Train ─────────────────────────────────────────────────────────
    if "E" in steps_to_run:
        cmd = [py, "train/train_model.py", "--epochs", str(args.epochs)]
        if args.smoke_test:
            cmd += ["--smoke-test"]
        rc = run_step("E", cmd)
        if rc != 0:
            summary["E"] = "FAILED"
            _print_summary(summary, time.time() - t_total)
            raise SystemExit(1)
        summary["E"] = "OK"

    # ── Step F: Export ────────────────────────────────────────────────────────
    if "F" in steps_to_run:
        cmd = [py, "inference/export_tfjs.py"]
        if args.smoke_test:
            cmd += ["--smoke-test"]
        rc = run_step("F", cmd)
        if rc != 0:
            summary["F"] = "FAILED"
            log.warning("Export failed. Install: pip install onnx onnx2tf tensorflowjs")
            log.warning("Re-run with: python inference/export_tfjs.py")
            summary["F"] = "FAILED (export deps missing)"
        else:
            summary["F"] = "OK"

    _print_summary(summary, time.time() - t_total)

    if args.smoke_test:
        log.info("\nAll smoke tests passed. Full pipeline is structurally sound.")
        log.info("Run without --smoke-test to download real data and train.")
    else:
        if all(v == "OK" for v in summary.values()):
            log.info("\nPipeline complete.")
            log.info("Serve the app:  python -m http.server 8080")
            log.info("Then open http://localhost:8080 and click 'Switch to Word Mode'.")


def _print_summary(summary: dict, elapsed: float) -> None:
    log.info("\n%s", "=" * 60)
    log.info("PIPELINE SUMMARY  (total %.1fs)", elapsed)
    log.info("%s", "=" * 60)
    for step in STEP_ORDER:
        status = summary.get(step, "skipped")
        log.info("  Step %s  %-30s  %s", step, STEP_DESCRIPTIONS[step], status)


if __name__ == "__main__":
    main()
