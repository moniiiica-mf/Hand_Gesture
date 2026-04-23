"""
Step E — GRU sequence classifier training.

Loads the numpy arrays produced by features/extract_sequences.py and
trains a two-layer GRU that maps (SEQ_LEN, FEAT_DIM) → 5-class softmax.

Outputs (in train/checkpoints/):
    best_model.pt          — PyTorch state-dict of the best val-loss epoch
    train_history.json     — loss / accuracy per epoch
    metrics.json           — final test accuracy, per-class F1, confusion matrix
    confusion_matrix.png   — visual confusion matrix

Usage:
    python train/train_model.py \
        --data-dir  data/processed \
        --ckpt-dir  train/checkpoints \
        [--epochs 60] [--batch 32] [--lr 1e-3] [--hidden 128] \
        [--dropout 0.3] [--patience 10] [--seed 42]

Requires: torch, scikit-learn, matplotlib (optional for the plot)
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


# ── Model definition ──────────────────────────────────────────────────────────

def build_model(feat_dim: int, num_classes: int, hidden: int, dropout: float):
    """Return a two-layer GRU classifier (lazy import of torch)."""
    import torch
    import torch.nn as nn

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=feat_dim,
                hidden_size=hidden,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, x):
            # x: (B, T, F) → gru out: (B, T, H), h_n: (layers, B, H)
            _, h_n = self.gru(x)
            last_hidden = h_n[-1]          # (B, H) — last layer's final state
            return self.head(last_hidden)  # (B, C)

    return GRUClassifier()


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_split(data_dir: Path, split: str):
    """Load (X, y) arrays; return (None, None) if files are missing."""
    xp = data_dir / f"X_{split}.npy"
    yp = data_dir / f"y_{split}.npy"
    if not xp.exists() or not yp.exists():
        return None, None
    X = np.load(str(xp)).astype(np.float32)
    y = np.load(str(yp)).astype(np.int64)
    return X, y


def make_loader(X, y, batch_size: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_class_weights(y, num_classes: int):
    """Inverse-frequency class weights for imbalanced datasets."""
    import torch

    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)  # avoid division by zero
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= num_classes   # rescale so mean weight ≈ 1
    return torch.tensor(weights, dtype=torch.float32)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    import torch

    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        n += len(y_batch)
    return total_loss / n, correct / n


@np.errstate(divide="ignore", invalid="ignore")
def eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            n += len(y_batch)
    return total_loss / n, correct / n


def predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_labels)


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def save_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping confusion matrix plot")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix (test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    log.info("Confusion matrix → %s", path)


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
    parser.add_argument("--patience",  type=int,   default=10,
                        help="Early-stopping patience (val loss)")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    import torch

    # ── Setup ─────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load label map ─────────────────────────────────────────────────────────
    id_to_label_path = data_dir / "id_to_label.json"
    if not id_to_label_path.exists():
        log.error("id_to_label.json missing — run extract_sequences.py first")
        raise SystemExit(1)
    id_to_label: dict = json.loads(id_to_label_path.read_text())
    num_classes = len(id_to_label)
    class_labels = [id_to_label[str(i)] for i in range(num_classes)]
    log.info("Classes (%d): %s", num_classes, class_labels)

    # ── Load data ──────────────────────────────────────────────────────────────
    X_train, y_train = load_split(data_dir, "train")
    X_val,   y_val   = load_split(data_dir, "val")
    X_test,  y_test  = load_split(data_dir, "test")

    if X_train is None or len(X_train) == 0:
        log.error("No training data found. Run download_clips.py + extract_sequences.py first.")
        raise SystemExit(1)

    feat_dim = X_train.shape[2]
    seq_len  = X_train.shape[1]
    log.info("Train: %d  Val: %d  Test: %d  |  shape=(_, %d, %d)",
             len(X_train),
             len(X_val) if X_val is not None else 0,
             len(X_test) if X_test is not None else 0,
             seq_len, feat_dim)

    train_loader = make_loader(X_train, y_train, args.batch, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   args.batch, shuffle=False) if X_val is not None else None
    test_loader  = make_loader(X_test,  y_test,  args.batch, shuffle=False) if X_test is not None else None

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(feat_dim, num_classes, args.hidden, args.dropout).to(device)
    log.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    class_weights = compute_class_weights(y_train, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=False
    )

    # ── Training ───────────────────────────────────────────────────────────────
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

            log.info("Epoch %3d/%d  tr_loss=%.4f  tr_acc=%.3f  "
                     "va_loss=%.4f  va_acc=%.3f  %s",
                     epoch, args.epochs, tr_loss, tr_acc, va_loss, va_acc,
                     "*" if improved else "")

            if patience_count >= args.patience:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                break
        else:
            # No validation split — save every epoch
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            log.info("Epoch %3d/%d  tr_loss=%.4f  tr_acc=%.3f",
                     epoch, args.epochs, tr_loss, tr_acc)

    elapsed = round(time.time() - t0, 1)
    log.info("Training finished in %.1fs", elapsed)

    # ── Evaluation on test set ─────────────────────────────────────────────────
    metrics: dict = {"elapsed_seconds": elapsed}

    if test_loader is not None:
        # Load best checkpoint
        best_ckpt = ckpt_dir / "best_model.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(str(best_ckpt), map_location=device))

        from sklearn.metrics import (
            accuracy_score, f1_score, confusion_matrix, classification_report
        )

        preds, labels = predict_all(model, test_loader, device)
        acc  = float(accuracy_score(labels, preds))
        f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
        f1_per   = f1_score(labels, preds, average=None, zero_division=0).tolist()
        cm       = confusion_matrix(labels, preds, labels=list(range(num_classes)))

        metrics.update({
            "test_accuracy":  round(acc, 4),
            "f1_macro":       round(f1_macro, 4),
            "f1_per_class":   {class_labels[i]: round(f1_per[i], 4)
                               for i in range(len(class_labels))},
            "confusion_matrix": cm.tolist(),
        })

        log.info("\n── Test results ────────────────────────────────────")
        log.info("Accuracy: %.3f  |  F1-macro: %.3f", acc, f1_macro)
        log.info("\n%s", classification_report(labels, preds,
                                              target_names=class_labels, zero_division=0))

        save_confusion_matrix(cm, class_labels, ckpt_dir / "confusion_matrix.png")
    else:
        log.warning("No test data — skipping evaluation")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    (ckpt_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("History  → %s", ckpt_dir / "train_history.json")
    log.info("Metrics  → %s", ckpt_dir / "metrics.json")
    log.info("Checkpoint → %s", ckpt_dir / "best_model.pt")


if __name__ == "__main__":
    main()
