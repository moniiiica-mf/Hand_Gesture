"""
Step E — GRU sequence classifier training.

Loads the numpy arrays produced by features/extract_sequences.py and
trains a two-layer GRU that maps (SEQ_LEN, FEAT_DIM) → 5-class softmax.

Outputs in --ckpt-dir (default train/checkpoints/):
    best_model.pt          PyTorch state-dict (best val-loss epoch)
    train_history.json     loss / accuracy per epoch
    metrics.json           test accuracy, per-class F1, confusion matrix
    confusion_matrix.png   (requires matplotlib)

Usage:
    python train/train_model.py \
        --data-dir data/processed \
        [--epochs 60] [--batch 32] [--lr 1e-3] [--hidden 128]

    # Smoke test — 3 epochs, tiny batch, skips test evaluation
    python train/train_model.py --smoke-test

Exit codes:
    0  Training completed (or smoke test passed)
    1  Missing data / dependency
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(feat_dim: int, num_classes: int, hidden: int, dropout: float):
    import torch.nn as nn

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                feat_dim, hidden, num_layers=2,
                batch_first=True, dropout=dropout, bidirectional=False,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, x):
            _, h_n = self.gru(x)
            return self.head(h_n[-1])

    return GRUClassifier()


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_split(data_dir: Path, split: str):
    xp, yp = data_dir / f"X_{split}.npy", data_dir / f"y_{split}.npy"
    if not xp.exists() or not yp.exists():
        return None, None
    return np.load(str(xp)).astype(np.float32), np.load(str(yp)).astype(np.int64)


def make_loader(X, y, batch_size: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def class_weights(y, num_classes: int):
    import torch
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    w = num_classes / counts
    return torch.tensor(w / w.sum() * num_classes, dtype=torch.float32)


# ── Training steps ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    import torch
    model.train()
    total_loss = correct = n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


def eval_epoch(model, loader, criterion, device):
    import torch
    model.eval()
    total_loss = correct = n = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def predict_all(model, loader, device):
    import torch
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


# ── Confusion matrix plot ──────────────────────────────────────────────────────

def save_cm(cm, label_names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(xticks=range(len(label_names)), yticks=range(len(label_names)),
               xticklabels=label_names, yticklabels=label_names,
               xlabel="Predicted", ylabel="True", title="Confusion Matrix (test)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(str(path), dpi=120)
        plt.close(fig)
        log.info("Confusion matrix → %s", path)
    except ImportError:
        log.warning("matplotlib not installed — skipping confusion matrix plot")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU word-sign classifier")
    parser.add_argument("--data-dir",  default="data/processed")
    parser.add_argument("--ckpt-dir",  default="train/checkpoints")
    parser.add_argument("--epochs",    type=int,   default=60)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--hidden",    type=int,   default=128)
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--patience",  type=int,   default=10)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="3 epochs, batch 4, skip full evaluation")
    args = parser.parse_args()

    import torch

    if args.smoke_test:
        log.info("SMOKE TEST: 3 epochs, batch=4")
        args.epochs = 3
        args.batch  = 4
        args.patience = 999

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    id2label_path = data_dir / "id_to_label.json"
    if not id2label_path.exists():
        log.error("id_to_label.json missing — run extract_sequences.py first")
        raise SystemExit(1)
    id_to_label: dict = json.loads(id2label_path.read_text())
    num_classes = len(id_to_label)
    class_labels = [id_to_label[str(i)] for i in range(num_classes)]
    log.info("Classes (%d): %s", num_classes, class_labels)

    X_train, y_train = load_split(data_dir, "train")
    X_val,   y_val   = load_split(data_dir, "val")
    X_test,  y_test  = load_split(data_dir, "test")

    if X_train is None or len(X_train) == 0:
        log.error("No training data found — run download_clips.py + extract_sequences.py first")
        raise SystemExit(1)

    log.info("Train: %d  Val: %d  Test: %d  |  shape=(_, %d, %d)",
             len(X_train),
             len(X_val) if X_val is not None else 0,
             len(X_test) if X_test is not None else 0,
             X_train.shape[1], X_train.shape[2])

    train_loader = make_loader(X_train, y_train, args.batch, shuffle=True)
    val_loader   = make_loader(X_val, y_val, args.batch, False) if X_val is not None else None
    test_loader  = make_loader(X_test, y_test, args.batch, False) if X_test is not None else None

    model = build_model(X_train.shape[2], num_classes, args.hidden, args.dropout).to(device)
    log.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    cw = class_weights(y_train, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_count = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))

        if val_loader is not None:
            va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
            history["val_loss"].append(round(va_loss, 4))
            history["val_acc"].append(round(va_acc, 4))
            scheduler.step(va_loss)
            improved = va_loss < best_val_loss
            if improved:
                best_val_loss = va_loss
                torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
                patience_count = 0
            else:
                patience_count += 1
            log.info("Epoch %3d/%d  tr=%.4f/%.3f  va=%.4f/%.3f  %s",
                     epoch, args.epochs, tr_loss, tr_acc, va_loss, va_acc,
                     "*" if improved else f"(pat {patience_count}/{args.patience})")
            if patience_count >= args.patience:
                log.info("Early stopping at epoch %d", epoch)
                break
        else:
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            log.info("Epoch %3d/%d  tr=%.4f/%.3f", epoch, args.epochs, tr_loss, tr_acc)

    elapsed = round(time.time() - t0, 1)
    (ckpt_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    log.info("Training done in %.1fs. History → %s", elapsed, ckpt_dir / "train_history.json")

    metrics: dict = {"elapsed_seconds": elapsed}

    if test_loader is not None and not args.smoke_test:
        best_ckpt = ckpt_dir / "best_model.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(str(best_ckpt), map_location=device))

        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        preds, labels = predict_all(model, test_loader, device)
        acc  = float(accuracy_score(labels, preds))
        f1m  = float(f1_score(labels, preds, average="macro", zero_division=0))
        f1pc = f1_score(labels, preds, average=None, zero_division=0).tolist()
        cm   = confusion_matrix(labels, preds, labels=list(range(num_classes)))

        metrics.update({
            "test_accuracy":  round(acc, 4),
            "f1_macro":       round(f1m, 4),
            "f1_per_class":   {class_labels[i]: round(f1pc[i], 4)
                               for i in range(len(class_labels))},
            "confusion_matrix": cm.tolist(),
        })
        log.info("\n── Test results ─────────────────────────────────────")
        log.info("Accuracy: %.3f  |  F1-macro: %.3f", acc, f1m)
        log.info("\n%s", classification_report(labels, preds,
                                              target_names=class_labels, zero_division=0))
        save_cm(cm, class_labels, ckpt_dir / "confusion_matrix.png")

    elif args.smoke_test:
        log.info("SMOKE TEST PASSED — GRU training pipeline is functional.")

    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Metrics → %s", ckpt_dir / "metrics.json")
    log.info("Checkpoint → %s", ckpt_dir / "best_model.pt")


if __name__ == "__main__":
    main()
