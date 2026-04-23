"""
Step B — Dataset manifest preparation.

Reads MS-ASL metadata, resolves labels for the 5-word vocabulary,
and writes a unified CSV manifest with one row per clip.

Vocabulary (class_id → label):
  0  hello          MS-ASL label   0
  1  how            MS-ASL label  59
  2  how_are_you    MS-ASL label 274  (compound; "are" alone is NOT in MS-ASL)
  3  you            MS-ASL label  68
  4  today          MS-ASL label 131

NOTE on "are":
  The word "are" does not appear as a standalone gloss in MSASL_classes.json.
  The closest available sign is the compound "how are you" (label 274).
  This script uses that compound as class 2 ("how_are_you").

Usage:
    python data/prepare_manifest.py \
        --msasl-dir MS-ASL \
        --out data/processed/msasl_5word_manifest.csv

    # Smoke test — first 3 clips per class per split, skip threshold check
    python data/prepare_manifest.py --smoke-test 3
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Vocabulary ─────────────────────────────────────────────────────────────────
VOCAB: dict[str, str] = {
    "hello":       "hello",
    "how":         "how",
    "how_are_you": "how are you",   # "are" alone is absent from MS-ASL
    "you":         "you",
    "today":       "today",
}
CLASS_ORDER = ["hello", "how", "how_are_you", "you", "today"]


def load_class_map(msasl_dir: Path) -> dict[str, int]:
    """Return {gloss_string: msasl_label_id} for our vocabulary."""
    with open(msasl_dir / "MSASL_classes.json") as f:
        all_classes = json.load(f)
    gloss_to_msasl = {g: i for i, g in enumerate(all_classes)}

    result: dict[str, int] = {}
    for local_label, gloss in VOCAB.items():
        if gloss not in gloss_to_msasl:
            log.error("Gloss '%s' not found in MSASL_classes.json", gloss)
            sys.exit(1)
        result[gloss] = gloss_to_msasl[gloss]
        log.info("  %-15s → MS-ASL label %3d  ('%s')",
                 local_label, gloss_to_msasl[gloss], gloss)
    return result


def build_manifest(
    msasl_dir: Path,
    gloss_to_msasl: dict[str, int],
    smoke_n: int = 0,
) -> list[dict]:
    """Collect matching entries across train/val/test splits."""
    gloss_to_local = {gloss: local for local, gloss in VOCAB.items()}
    msasl_id_to_local = {msasl_id: gloss_to_local[gloss]
                         for gloss, msasl_id in gloss_to_msasl.items()}
    local_to_id = {label: i for i, label in enumerate(CLASS_ORDER)}
    rows: list[dict] = []

    for split in ("train", "val", "test"):
        path = msasl_dir / f"MSASL_{split}.json"
        with open(path) as f:
            entries = json.load(f)

        class_counts: dict[str, int] = defaultdict(int)

        for e in entries:
            if e["label"] not in msasl_id_to_local:
                continue
            local_label = msasl_id_to_local[e["label"]]

            if smoke_n and class_counts[local_label] >= smoke_n:
                continue
            class_counts[local_label] += 1

            rows.append({
                "split":      split,
                "class_id":   local_to_id[local_label],
                "label":      local_label,
                "url":        e["url"],
                "start_time": round(e["start_time"], 4),
                "end_time":   round(e["end_time"], 4),
                "duration":   round(e["end_time"] - e["start_time"], 4),
                "signer_id":  e["signer_id"],
                "fps":        e.get("fps", 30),
                "width":      e.get("width", 640),
                "height":     e.get("height", 360),
                "box":        json.dumps(e.get("box", [])),
            })
    return rows


def print_stats(rows: list[dict], min_train_clips: int) -> bool:
    """Print per-class counts; return True if all train classes meet threshold."""
    log.info("\n── Dataset statistics ──────────────────────────────")
    by_split: dict[str, list] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    all_ok = True
    for split in ("train", "val", "test"):
        entries = by_split[split]
        by_class: dict[str, int] = defaultdict(int)
        for r in entries:
            by_class[r["label"]] += 1
        log.info("%s  (%d clips total)", split.upper(), len(entries))
        for label in CLASS_ORDER:
            n = by_class.get(label, 0)
            flag = ""
            if split == "train" and n < min_train_clips:
                flag = f"  *** BELOW THRESHOLD ({min_train_clips})"
                all_ok = False
            log.info("    %-15s %3d%s", label, n, flag)

    log.info("\nGrand total: %d clips across %d classes", len(rows), len(CLASS_ORDER))
    durations = [r["duration"] for r in rows]
    if durations:
        log.info(
            "Clip duration — min: %.1fs  mean: %.1fs  max: %.1fs",
            min(durations), sum(durations) / len(durations), max(durations),
        )
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MS-ASL 5-word manifest CSV")
    parser.add_argument("--msasl-dir",  default="MS-ASL")
    parser.add_argument("--out",        default="data/processed/msasl_5word_manifest.csv")
    parser.add_argument("--min-train-clips", type=int, default=15,
                        help="Warn if any train class has fewer manifest rows than this")
    parser.add_argument("--smoke-test", type=int, default=0, metavar="N",
                        help="Keep only the first N clips per class per split (pipeline test)")
    args = parser.parse_args()

    msasl_dir = Path(args.msasl_dir)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("─── Word availability check ───────────────────────")
    log.info("  hello        → MS-ASL label   0  OK")
    log.info("  how          → MS-ASL label  59  OK")
    log.info("  are          → NOT FOUND in MS-ASL  → substituted with 'how are you' (label 274)")
    log.info("  you          → MS-ASL label  68  OK")
    log.info("  today        → MS-ASL label 131  OK")
    log.info("")
    log.info("Resolved vocabulary:")
    gloss_to_msasl = load_class_map(msasl_dir)

    if args.smoke_test:
        log.info("SMOKE TEST: capping at %d clips per class per split", args.smoke_test)

    rows = build_manifest(msasl_dir, gloss_to_msasl, smoke_n=args.smoke_test)

    fieldnames = ["split", "class_id", "label", "url", "start_time", "end_time",
                  "duration", "signer_id", "fps", "width", "height", "box"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Manifest written → %s  (%d rows)", out_path, len(rows))

    all_ok = print_stats(rows, args.min_train_clips)
    if not all_ok and not args.smoke_test:
        log.warning("Some train classes are below --min-train-clips=%d.", args.min_train_clips)
        log.warning("how_are_you has only 19 train clips in MS-ASL — dead links may reduce this.")
        log.warning("The download step will enforce the threshold and stop if unmet.")

    id2label = {i: l for i, l in enumerate(CLASS_ORDER)}
    label2id = {l: i for i, l in id2label.items()}
    processed = out_path.parent
    with open(processed / "id_to_label.json", "w") as f:
        json.dump(id2label, f, indent=2)
    with open(processed / "label_to_id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    log.info("Label maps → %s", processed)


if __name__ == "__main__":
    main()
