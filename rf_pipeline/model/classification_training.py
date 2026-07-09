from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ULTRALYTICS_CLASSIFIERS = {
    "yolo11n_cls": "yolo11n-cls.pt",
    "yolo26n_cls": "yolo26n-cls.pt",
}

TORCHVISION_CLASSIFIERS = {
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_v2_s",
    "convnext_tiny",
}

CLASSIFICATION_MODEL_CHOICES = tuple(sorted((*ULTRALYTICS_CLASSIFIERS, *TORCHVISION_CLASSIFIERS)))


@dataclass
class TorchvisionTrainResult:
    save_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_val_acc: float


def build_torchvision_classifier(model_name: str, num_classes: int, pretrained: bool = False):
    """Build a torchvision classifier and replace its final head."""
    import torch.nn as nn
    from torchvision import models

    weights = "DEFAULT" if pretrained else None

    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported torchvision classifier: {model_name}")


def _resolve_device(device: str | None):
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _make_loaders(data_dir: Path, imgsz: int, batch: int, workers: int):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    train_tf = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf) if (data_dir / "test").exists() else None

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = (
        DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
        if test_ds is not None
        else None
    )
    return train_ds, train_loader, val_loader, test_loader


def _evaluate(model, loader, device) -> tuple[float, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


def train_torchvision_classifier(
    model_name: str,
    data_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None,
    workers: int,
    project: str | Path,
    run_name: str,
    seed: int,
    pretrained: bool = False,
    lr: float = 1e-3,
) -> TorchvisionTrainResult:
    """Train a torchvision classifier on ImageFolder train/val/test splits."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    device_obj = _resolve_device(device)
    data_dir = data_dir.resolve()
    train_ds, train_loader, val_loader, test_loader = _make_loaders(data_dir, imgsz, batch, workers)

    model = build_torchvision_classifier(model_name, len(train_ds.classes), pretrained=pretrained).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    save_dir = Path(project) / run_name
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = weights_dir / "best.pt"
    last_checkpoint = weights_dir / "last.pt"
    metrics_path = save_dir / "metrics.csv"

    best_val_acc = -1.0
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total = 0
            correct = 0
            for images, targets in train_loader:
                images = images.to(device_obj, non_blocking=True)
                targets = targets.to(device_obj, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * targets.size(0)
                correct += int((logits.argmax(dim=1) == targets).sum().item())
                total += int(targets.size(0))

            scheduler.step()
            train_loss = total_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            val_loss, val_acc = _evaluate(model, val_loader, device_obj)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                }
            )
            f.flush()
            print(
                f"epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            checkpoint = {
                "model_name": model_name,
                "classes": train_ds.classes,
                "state_dict": model.state_dict(),
                "imgsz": imgsz,
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(checkpoint, last_checkpoint)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(checkpoint, best_checkpoint)

    if test_loader is not None:
        test_loss, test_acc = _evaluate(model, test_loader, device_obj)
        with (save_dir / "test_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["test_loss", "test_acc"])
            writer.writeheader()
            writer.writerow({"test_loss": f"{test_loss:.6f}", "test_acc": f"{test_acc:.6f}"})
        print(f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    return TorchvisionTrainResult(
        save_dir=save_dir,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        best_val_acc=best_val_acc,
    )
