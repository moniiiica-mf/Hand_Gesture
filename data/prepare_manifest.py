"""
Step B — Dataset manifest preparation.

Reads MS-ASL metadata, resolves labels for the 5-word vocabulary,
and writes a unified CSV manifest with one row per clip.

Vocabulary (class_id → label):
  0  hello
  1  how
  2  how_are_you   (compound sign; MS-ASL class "how are you")
  3  you
  4  today

Usage:
    python data/prepare_manifest.py \
        --msasl-dir MS-ASL \
        --out data/processed/msasl_5word_manifest.csv
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

# ── Vocabulary ────────────────────────────────────────────────────────────────
# Maps our local class label → MS-ASL gloss string (must match MSASL_classes.json exactly)
VOCAB: dict[str, str] = {
    "hello":       "hello",
    "how":         "how",
    "how_are_you": "how are you",
    "you":         "you",
    "today":       "today",
}
# Ordered list defines class IDs 0-4
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
        log.info("  %-15s → MS-ASL label %d ('%s')", local_label, gloss_to_msasl[gloss], gloss)
    return result


def build_manifest(msasl_dir: Path, gloss_to_msasl: dict[str, int]) -> list[dict]:
    """Collect matching entries across train/val/test splits."""
    # gloss_to_msasl maps gloss_string → msasl_id; invert to msasl_id → local_label
    gloss_to_local = {gloss: local for local, gloss in VOCAB.items()}
    msasl_id_to_local = {msasl_id: gloss_to_local[gloss]
                         for gloss, msasl_id in gloss_to_msasl.items()}
    local_to_id = {label: i for i, label in enumerate(CLASS_ORDER)}
    rows: list[dict] = []

    for split in ("train", "val", "test"):
        path = msasl_dir / f"MSASL_{split}.json"
        with open(path) as f:
            entries = json.load(f)

        for e in entries:
            if e["label"] not in msasl_id_to_local:
                continue
            local_label = msasl_id_to_local[e["label"]]
            rows.append(
                {
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
                }
            )
    return rows


def print_stats(rows: list[dict]) -> None:
    log.info("\n── Dataset statistics ──────────────────────────────")
    by_split: dict[str, list] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    for split in ("train", "val", "test"):
        entries = by_split[split]
        by_class: dict[str, int] = defaultdict(int)
        for r in entries:
            by_class[r["label"]] += 1
        total = len(entries)
        log.info("%s  (%d clips total)", split.upper(), total)
        for label in CLASS_ORDER:
            log.info("    %-15s %d", label, by_class.get(label, 0))

    log.info("\nGrand total: %d clips across %d classes", len(rows), len(CLASS_ORDER))

    durations = [r["duration"] for r in rows]
    if durations:
        log.info(
            "Clip duration — min: %.1fs  mean: %.1fs  max: %.1fs",
            min(durations), sum(durations) / len(durations), max(durations),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MS-ASL 5-word manifest CSV")
    parser.add_argument("--msasl-dir",  default="MS-ASL",
                        help="Path to MS-ASL metadata directory")
    parser.add_argument("--out",        default="data/processed/msasl_5word_manifest.csv",
                        help="Output manifest CSV path")
    args = parser.parse_args()

    msasl_dir = Path(args.msasl_dir)
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading MS-ASL class map from %s", msasl_dir)
    gloss_to_msasl = load_class_map(msasl_dir)

    log.info("Building manifest …")
    rows = build_manifest(msasl_dir, gloss_to_msasl)

    # Write CSV
    fieldnames = ["split","class_id","label","url","start_time","end_time",
                  "duration","signer_id","fps","width","height","box"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Manifest written → %s  (%d rows)", out_path, len(rows))
    print_stats(rows)

    # Also write label maps
    id2label = {i: l for i, l in enumerate(CLASS_ORDER)}
    label2id = {l: i for i, l in id2label.items()}
    processed = Path(args.out).parent
    with open(processed / "id_to_label.json", "w") as f:
        json.dump(id2label, f, indent=2)
    with open(processed / "label_to_id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    log.info("Label maps written → %s", processed)


if __name__ == "__main__":
    main()
