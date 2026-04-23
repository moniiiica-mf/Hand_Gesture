"""
Step F helper — Export trained PyTorch GRU model to TensorFlow.js format.

Converts train/checkpoints/best_model.pt → model/word_gru/model.json
so that index.html can load it with tf.loadLayersModel().

Conversion path:  PyTorch → ONNX → TensorFlow SavedModel → TF.js

Dependencies (install once):
    pip install onnx onnx2tf tensorflowjs

Usage:
    python inference/export_tfjs.py \
        [--ckpt  train/checkpoints/best_model.pt] \
        [--out   model/word_gru] \
        [--seq-len 30] [--feat-dim 63] [--hidden 128] [--num-classes 5]
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def check_deps() -> None:
    missing = []
    for pkg in ("onnx", "onnx2tf", "tensorflowjs"):
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error("Missing packages: %s", missing)
        log.error("Install with: pip install %s", " ".join(missing))
        raise SystemExit(1)


def export_onnx(ckpt_path: Path, onnx_path: Path,
                seq_len: int, feat_dim: int, hidden: int, num_classes: int) -> None:
    import torch
    import torch.nn as nn

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(feat_dim, hidden, num_layers=2,
                              batch_first=True, dropout=0.0, bidirectional=False)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, x):
            _, h_n = self.gru(x)
            return self.head(h_n[-1])

    model = GRUClassifier()
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, seq_len, feat_dim)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=13,
    )
    log.info("ONNX exported → %s", onnx_path)


def onnx_to_savedmodel(onnx_path: Path, sm_dir: Path) -> None:
    rc = subprocess.run(
        ["onnx2tf", "-i", str(onnx_path), "-o", str(sm_dir),
         "--non_verbose"],
        capture_output=False,
    ).returncode
    if rc != 0:
        log.error("onnx2tf failed (rc=%d)", rc)
        raise SystemExit(1)
    log.info("SavedModel written → %s", sm_dir)


def savedmodel_to_tfjs(sm_dir: Path, out_dir: Path) -> None:
    import tensorflowjs.converters as conv
    conv.convert_tf_saved_model(
        str(sm_dir),
        str(out_dir),
    )
    log.info("TF.js model written → %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GRU model to TF.js")
    parser.add_argument("--ckpt",        default="train/checkpoints/best_model.pt")
    parser.add_argument("--out",         default="model/word_gru")
    parser.add_argument("--seq-len",     type=int, default=30)
    parser.add_argument("--feat-dim",    type=int, default=63)
    parser.add_argument("--hidden",      type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=5)
    args = parser.parse_args()

    check_deps()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s — run train/train_model.py first", ckpt_path)
        raise SystemExit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path    = Path(tmp)
        onnx_path   = tmp_path / "model.onnx"
        sm_dir      = tmp_path / "saved_model"

        log.info("Step 1/3 — Export to ONNX")
        export_onnx(ckpt_path, onnx_path,
                    args.seq_len, args.feat_dim, args.hidden, args.num_classes)

        log.info("Step 2/3 — Convert ONNX → TF SavedModel")
        onnx_to_savedmodel(onnx_path, sm_dir)

        log.info("Step 3/3 — Convert SavedModel → TF.js")
        savedmodel_to_tfjs(sm_dir, out_dir)

    # Write class metadata alongside the model
    classes = ["hello", "how", "how_are_you", "you", "today"]
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=2))

    log.info("\nDone. Serve index.html and the model will load from: %s/model.json", out_dir)
    log.info("(Use a local HTTP server: python -m http.server 8080)")


if __name__ == "__main__":
    main()
