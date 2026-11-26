"""
Face Emotion Recognition on FER2013 with a CNN (PyTorch).

Expected dataset layout (Kaggle "msambare/fer2013" image-folder version):

    fer2013/
        train/
            angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
        test/
            angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/

Run:
    python train.py --data_dir fer2013 --epochs 40

Quick smoke test first (a few minutes):
    python train.py --data_dir fer2013 --epochs 2
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = (0.5,), (0.5,)


class EmotionCNN(nn.Module):
    """3 conv blocks (64 -> 128 -> 256) + classifier. Input: 1x48x48 grayscale."""

    def __init__(self, num_classes: int = 7):
        super().__init__()

        def block(c_in, c_out, drop):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(drop),
            )

        self.features = nn.Sequential(
            block(1, 64, 0.2),     # 48 -> 24
            block(64, 128, 0.25),  # 24 -> 12
            block(128, 256, 0.3),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(data_dir: Path, batch_size: int, val_frac: float, workers: int, pin: bool):
    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(48, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Same folder twice so train gets augmentation and validation does not.
    full_train = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    full_val = datasets.ImageFolder(data_dir / "train", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    n = len(full_train)
    idx = np.random.RandomState(42).permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val].tolist(), idx[n_val:].tolist()

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)

    kw = dict(num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    targets = np.array(full_train.targets)[train_idx]
    counts = np.bincount(targets, minlength=len(full_train.classes))
    return train_loader, val_loader, test_loader, full_train.classes, counts


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
    return total_loss / seen, correct / seen


def plot_curves(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, classes, path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test, row-normalised)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="fer2013", help="folder containing train/ and test/")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3, help="peak LR for OneCycle")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10, help="early-stop patience (epochs)")
    p.add_argument("--workers", type=int, default=2, help="set 0 if you get errors on Windows")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, classes, counts = get_loaders(
        Path(args.data_dir), args.batch_size, args.val_frac, args.workers, device.type == "cuda"
    )
    print("Classes:", classes)
    print("Train images per class:", dict(zip(classes, counts.tolist())))

    # Softened inverse-frequency weights: 'disgust' has very few images.
    weights = (counts.sum() / (len(classes) * counts)) ** 0.5
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=0.1,
    )

    model = EmotionCNN(len(classes)).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc, best_epoch = 0.0, 0
    ckpt_path = out_dir / "emotion_cnn.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        vl, va = run_epoch(model, val_loader, criterion, device)
        for k, v in zip(history, (tl, ta, vl, va)):
            history[k].append(v)
        flag = ""
        if va > best_acc:
            best_acc, best_epoch = va, epoch
            torch.save({"model": model.state_dict(), "classes": classes, "val_acc": va}, ckpt_path)
            flag = "  <- best, saved"
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.3f} acc {ta:.3f} | "
              f"val loss {vl:.3f} acc {va:.3f} | {time.time() - t0:.0f}s{flag}")
        if epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping early.")
            break

    # Final evaluation on the held-out test set with the best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    test_acc = float((preds == labels).mean())

    report = classification_report(labels, preds, target_names=classes, digits=3)
    print(f"\nBest val acc {best_acc:.3f} (epoch {best_epoch}) | TEST acc {test_acc:.3f}\n")
    print(report)

    (out_dir / "classification_report.txt").write_text(f"Test accuracy: {test_acc:.4f}\n\n{report}")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_curves(history, out_dir / "training_curves.png")
    plot_confusion(confusion_matrix(labels, preds), classes, out_dir / "confusion_matrix.png")
    print(f"Saved model, curves, confusion matrix and report in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
