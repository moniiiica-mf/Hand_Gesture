"""
Step F helper — Export trained PyTorch GRU model to TensorFlow.js format.

Converts train/checkpoints/best_model.pt → model/word_gru/model.json
so that index.html can load it with tf.loadGraphModel().

Conversion path:  PyTorch → ONNX → TF SavedModel → TF.js

Dependencies (install once):
    pip install onnx onnx2tf tensorflowjs

Usage:
    python inference/export_tfjs.py \
        [--ckpt  train/checkpoints/best_model.pt] \
        [--out   model/word_gru] \
        [--seq-len 30] [--feat-dim 63] [--hidden 128] [--num-classes 5]

    # Smoke test — verify PyTorch→ONNX only; skip TF conversion
    python inference/export_tfjs.py --smoke-test

Exit codes:
    0  Export succeeded (or smoke test passed)
    1  Missing checkpoint / dependency
"""

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CLASSES = ["hello", "how", "how_are_you", "you", "today"]


def _gru_model(feat_dim, num_classes, hidden):
    """Reconstruct the same GRU architecture as train_model.py."""
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

    return GRUClassifier()


def export_onnx(ckpt_path, onnx_path, seq_len, feat_dim, hidden, num_classes):
    import torch
    model = _gru_model(feat_dim, num_classes, hidden)
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model.eval()
    dummy = torch.zeros(1, seq_len, feat_dim)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=13,
    )
    log.info("ONNX exported → %s", onnx_path)


def onnx_to_savedmodel(onnx_path, sm_dir):
    rc = subprocess.run(
        ["onnx2tf", "-i", str(onnx_path), "-o", str(sm_dir), "--non_verbose"],
    ).returncode
    if rc != 0:
        log.error("onnx2tf failed (rc=%d)", rc)
        raise SystemExit(1)
    log.info("SavedModel → %s", sm_dir)


def savedmodel_to_tfjs(sm_dir, out_dir):
    import tensorflowjs.converters as conv
    conv.convert_tf_saved_model(str(sm_dir), str(out_dir))
    log.info("TF.js model → %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GRU checkpoint to TF.js")
    parser.add_argument("--ckpt",        default="train/checkpoints/best_model.pt")
    parser.add_argument("--out",         default="model/word_gru")
    parser.add_argument("--seq-len",     type=int, default=30)
    parser.add_argument("--feat-dim",    type=int, default=63)
    parser.add_argument("--hidden",      type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Only verify PyTorch→ONNX; skip TF conversion")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s — run train/train_model.py first", ckpt_path)
        raise SystemExit(1)

    try:
        import torch
    except ImportError:
        log.error("torch not installed")
        raise SystemExit(1)

    if args.smoke_test:
        log.info("SMOKE TEST: verifying PyTorch → ONNX only")
        with tempfile.TemporaryDirectory() as tmp:
            export_onnx(ckpt_path, Path(tmp) / "model.onnx",
                        args.seq_len, args.feat_dim, args.hidden, args.num_classes)
        log.info("SMOKE TEST PASSED — PyTorch model loads and exports to ONNX.")
        return

    for pkg in ("onnx", "onnx2tf", "tensorflowjs"):
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            log.error("Missing: %s  →  pip install onnx onnx2tf tensorflowjs", pkg)
            raise SystemExit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path  = Path(tmp)
        onnx_path = tmp_path / "model.onnx"
        sm_dir    = tmp_path / "saved_model"

        log.info("1/3  PyTorch → ONNX")
        export_onnx(ckpt_path, onnx_path, args.seq_len, args.feat_dim, args.hidden, args.num_classes)

        log.info("2/3  ONNX → TF SavedModel")
        onnx_to_savedmodel(onnx_path, sm_dir)

        log.info("3/3  SavedModel → TF.js")
        savedmodel_to_tfjs(sm_dir, out_dir)

    (out_dir / "classes.json").write_text(json.dumps(CLASSES, indent=2))
    log.info("\nDone. Place %s/ at the repo root and serve with:", out_dir)
    log.info("  python -m http.server 8080")


if __name__ == "__main__":
    main()
