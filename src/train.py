"""
train.py
========
Training pipeline for Sentinel-2 multi-label land-cover classification.
Supports reproducible band ablation experiments (M1: Visible, M2: Visible+NIR, M3: Full Multispectral).

Key Guarantees:
  - 100% genuine dataset: curated_dataset_phase2_1.npz
  - Exact official split partitioning: Train (300), Validation (100), Test (100)
  - Zero test data leakage in model parameters or preprocessing statistics
  - Identical random seeds and patch partitions across all band experiments
  - Multi-label objective: BCEWithLogitsLoss with sigmoid activation
  - Early stopping on validation loss with checkpoint saving
"""

import os
import sys
import time
import random
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.model import (
    BAND_CONFIGS,
    MultispectralDataset,
    MultispectralCNN,
    build_model,
    count_parameters
)
from src.label_mapping import BROAD_CLASSES, NUM_CLASSES


def set_seed(seed: int = 42):
    """Sets reproducible random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset_splits(
    npz_path: str = "data/curated_dataset_phase2_1.npz"
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads genuine dataset and slices into official Train, Validation, and Test partitions.
    Guarantees strict zero-overlap and identical patch order.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset archive not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    images = data["images"]       # (500, 6, 120, 120)
    labels = data["labels"]       # (500, 7)
    patch_ids = data["patch_ids"] # (500,)
    splits = data["splits"]       # (500,)

    train_mask = (splits == "train")
    val_mask = (splits == "validation")
    test_mask = (splits == "test")

    return {
        "train": {
            "images": images[train_mask],
            "labels": labels[train_mask],
            "patch_ids": patch_ids[train_mask]
        },
        "validation": {
            "images": images[val_mask],
            "labels": labels[val_mask],
            "patch_ids": patch_ids[val_mask]
        },
        "test": {
            "images": images[test_mask],
            "labels": labels[test_mask],
            "patch_ids": patch_ids[test_mask]
        }
    }


def create_dataloaders(
    dataset_splits: Dict[str, Dict[str, np.ndarray]],
    channel_indices: List[int],
    batch_size: int = 16,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Constructs PyTorch DataLoaders for Train, Validation, and Test sets
    using the specified channel indices (e.g. M1, M2, M3).
    """
    train_ds = MultispectralDataset(
        dataset_splits["train"]["images"],
        dataset_splits["train"]["labels"],
        channel_indices=channel_indices
    )
    val_ds = MultispectralDataset(
        dataset_splits["validation"]["images"],
        dataset_splits["validation"]["labels"],
        channel_indices=channel_indices
    )
    test_ds = MultispectralDataset(
        dataset_splits["test"]["images"],
        dataset_splits["test"]["labels"],
        channel_indices=channel_indices
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluates model on a dataloader.
    Returns: (average_loss, macro_f1, probabilities, targets)
    """
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(imgs)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    probs_arr = np.vstack(all_probs)
    targets_arr = np.vstack(all_targets)

    # Compute Macro-F1 at specified threshold
    preds_bin = (probs_arr >= threshold).astype(int)
    macro_f1 = float(f1_score(targets_arr, preds_bin, average="macro", zero_division=0))

    return avg_loss, macro_f1, probs_arr, targets_arr


def train_single_model(
    model_name: str,
    band_config: Dict[str, any],
    dataset_splits: Dict[str, Dict[str, np.ndarray]],
    checkpoint_dir: str = "models",
    batch_size: int = 16,
    max_epochs: int = 35,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Trains a model for a specific band configuration with early stopping.
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    in_channels = band_config["in_channels"]
    channel_indices = band_config["indices"]

    if verbose:
        print("\n" + "=" * 70)
        print(f" TRAINING: {model_name} ({band_config['name']})")
        print(f" Bands: {band_config['bands']} (in_channels={in_channels})")
        print(f" Device: {device} | Seed: {seed} | Batch Size: {batch_size}")
        print("=" * 70)

    # Instantiate model
    model = MultispectralCNN(in_channels=in_channels, num_classes=NUM_CLASSES)
    model.to(device)
    param_info = count_parameters(model)
    if verbose:
        print(f"  Model Parameters: {param_info['trainable']:,} trainable")

    loaders = create_dataloaders(dataset_splits, channel_indices=channel_indices, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_macro_f1 = 0.0
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0

    history = []
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name.lower()}_best.pt")

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # --- Training step ---
        model.train()
        train_loss_total = 0.0
        for imgs, targets in loaders["train"]:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(imgs)

        train_loss = train_loss_total / len(loaders["train"].dataset)

        # --- Validation step ---
        val_loss, val_macro_f1, _, _ = evaluate_epoch(
            model, loaders["validation"], criterion, device, threshold=0.5
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - start_time
        history.append({
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "lr": current_lr,
            "elapsed_sec": round(elapsed, 2)
        })

        if verbose and (epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss):
            is_best_mark = " *" if val_loss < best_val_loss else ""
            print(f"  Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | LR: {current_lr:.2e}{is_best_mark}")

        # Check early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            # Save checkpoint immediately
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "band_config": band_config,
                "model_state_dict": best_state_dict,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "seed": seed,
                "classes": BROAD_CLASSES
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  [Early Stopping] No validation loss improvement for {patience} epochs. Stopping at epoch {epoch}.")
                break

    if verbose:
        print(f"  ✓ Best model saved: {checkpoint_path} (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}, Val Macro-F1: {best_val_macro_f1:.4f})")

    # Load best weights before returning
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model_name": model_name,
        "band_config": band_config,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": checkpoint_path,
        "history": pd.DataFrame(history),
        "model": model,
        "loaders": loaders
    }
