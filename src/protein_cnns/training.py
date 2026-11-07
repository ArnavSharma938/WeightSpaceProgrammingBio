from __future__ import annotations

import copy
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import DATA_DIR, RUNS_DIR, SEED, seed_everything, sha256, write_json
from .metrics import classification_metrics
from .model import ProteinCNN


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_task(task: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / f"{task}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/cnns.py prepare first")
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def one_hot(encoded: torch.Tensor) -> torch.Tensor:
    return functional.one_hot(encoded.long(), num_classes=21).float().transpose(1, 2).contiguous()


@torch.inference_mode()
def predict(model: nn.Module, sequences: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    output = []
    for offset in range(0, len(sequences), batch_size):
        encoded = torch.from_numpy(sequences[offset : offset + batch_size]).to(device)
        probabilities = torch.softmax(model(one_hot(encoded)), dim=1)[:, 1]
        output.append(probabilities.cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def train_task(
    task: str,
    device_name: str = "auto",
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
) -> dict:
    seed_everything(SEED)
    device = resolve_device(device_name)
    data = load_task(task)
    train_sequences = torch.from_numpy(data["train_sequences"])
    train_labels = torch.from_numpy(data["train_labels"])
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(train_sequences, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
    model = ProteinCNN().to(device)
    if model.parameter_count != 16_418:
        raise AssertionError(f"Unexpected parameter count: {model.parameter_count}")

    run_dir = RUNS_DIR / f"seed_{SEED}" / task
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_path = checkpoint_dir / "initial_state_dict.pt"
    torch.save(model.state_dict(), initial_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_auc = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for encoded, labels in loader:
            encoded = encoded.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(one_hot(encoded))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)

        validation_scores = predict(model, data["validation_sequences"], device)
        validation_metrics = classification_metrics(data["validation_labels"], validation_scores)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{task}: epoch {epoch}/{epochs}, loss={row['train_loss']:.4f}, "
                f"val_auc={row['validation_roc_auc']:.4f}, "
                f"val_bal_acc={row['validation_balanced_accuracy']:.4f}",
                flush=True,
            )
        auc = float(validation_metrics["roc_auc"])
        if auc > best_auc + 1e-8:
            best_auc = auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= 15 and epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    best_path = checkpoint_dir / "best_state_dict.pt"
    torch.save(model.state_dict(), best_path)
    train_scores = predict(model, data["train_sequences"], device)
    validation_scores = predict(model, data["validation_sequences"], device)
    summary = {
        "task": task,
        "seed": SEED,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "parameter_count": model.parameter_count,
        "hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "maximum_epochs": epochs,
            "early_stopping_patience": patience,
            "selection_metric": "validation ROC AUC",
        },
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "train_metrics": classification_metrics(data["train_labels"], train_scores),
        "validation_metrics": classification_metrics(data["validation_labels"], validation_scores),
        "initial_checkpoint_sha256": sha256(initial_path),
        "best_checkpoint_sha256": sha256(best_path),
    }
    write_json(run_dir / "training_history.json", history)
    write_json(run_dir / "training_summary.json", summary)
    return summary
