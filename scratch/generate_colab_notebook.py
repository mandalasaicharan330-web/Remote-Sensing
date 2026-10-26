"""
generate_colab_notebook.py
==========================
Generates a complete, Google Colab-compatible Jupyter Notebook (.ipynb)
for Sentinel-2 Multi-Label Land-Cover Classification (Phase 3).
"""

import os
import json

def build_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": []
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header
    add_md("""# Sentinel-2 Multi-Label Land-Cover Classification & Band Ablation (M1/M2/M3)
**Project:** Applied Geophysics & Remote Sensing Pair Programming Project  
**Author:** Antigravity IDE & Pair Programming Assistant  
**Dataset:** BigEarthNet v2.0 Curated Dataset (500 genuine Sentinel-2 Level-2A patches)  
**Objective:** Train and evaluate multi-label Convolutional Neural Networks on Sentinel-2 multispectral imagery and evaluate the scientific contribution of NIR (B08) and SWIR (B11, B12) over standard Visible RGB (B02, B03, B04).
""")

    # Setup
    add_md("""## 1. Environment Setup & Dependency Installation
If running in Google Colab, execute the cell below to install required packages (`pyarrow`, `zstandard`, etc.) and verify GPU acceleration.
""")
    add_code("""# Install dependencies if in Colab
import sys
if 'google.colab' in sys.modules:
    !pip install -q pyarrow zstandard

import os
import time
import json
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

print(f"PyTorch Version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Compute Device: {device}")
""")

    # Data loading
    add_md("""## 2. Dataset Loading (Curated Phase 2.1 Dataset)
The dataset consists of **500 genuine Sentinel-2 Level-2A patches** stored in `data/curated_dataset_phase2_1.npz`.

If running in Google Colab:
- **Option A:** Upload `curated_dataset_phase2_1.npz` directly via the file upload cell below.
- **Option B:** Mount your Google Drive where the file is stored.
- **Option C:** Load from local repository folder `data/` if the repo is cloned.
""")
    add_code("""# Locate dataset file
dataset_path = "data/curated_dataset_phase2_1.npz"

if not os.path.exists(dataset_path):
    if 'google.colab' in sys.modules:
        print("Curated dataset not found locally. Please upload 'curated_dataset_phase2_1.npz':")
        from google.colab import files
        uploaded = files.upload()
        for fn in uploaded.keys():
            if fn.endswith('.npz'):
                dataset_path = fn
                break
    else:
        raise FileNotFoundError(f"Could not find {dataset_path}. Please check working directory.")

print(f"Loading verified dataset from: {dataset_path}")
data = np.load(dataset_path, allow_pickle=True)

images = data["images"]       # (500, 6, 120, 120)
labels = data["labels"]       # (500, 7)
patch_ids = data["patch_ids"] # (500,)
splits = data["splits"]       # (500,)
class_names = [str(c) for c in data["class_names"]]
band_names = [str(b) for b in data["band_names"]]

print(f"Images shape: {images.shape} (dtype: {images.dtype})")
print(f"Labels shape: {labels.shape} (dtype: {labels.dtype})")
print(f"Reflectance bounds: [{images.min():.4f}, {images.max():.4f}]")
print(f"Target classes: {class_names}")
print(f"Spectral bands: {band_names}")
""")

    # Partitions
    add_md("""## 3. Dataset Partitions and Split Verification
The dataset strictly preserves the official BigEarthNet v2.0 split assignments:
- **Train:** 300 patches (60.0%)
- **Validation:** 100 patches (20.0%)
- **Test:** 100 patches (20.0%)
""")
    add_code("""# Partition masks
train_mask = (splits == "train")
val_mask = (splits == "validation")
test_mask = (splits == "test")

train_ids = set(patch_ids[train_mask])
val_ids = set(patch_ids[val_mask])
test_ids = set(patch_ids[test_mask])

print(f"Train Patches:      {len(train_ids)}")
print(f"Validation Patches: {len(val_ids)}")
print(f"Test Patches:       {len(test_ids)}")

assert len(train_ids & val_ids) == 0 and len(train_ids & test_ids) == 0 and len(val_ids & test_ids) == 0, \\
    "Error: Data leakage between splits!"
print("Zero data leakage verified across splits.")

# Class distributions
class_summary = []
for c_idx, c_name in enumerate(class_names):
    tr_cnt = int(np.sum(labels[train_mask, c_idx]))
    va_cnt = int(np.sum(labels[val_mask, c_idx]))
    te_cnt = int(np.sum(labels[test_mask, c_idx]))
    class_summary.append({
        "Class": c_name,
        "Train Count": tr_cnt,
        "Val Count": va_cnt,
        "Test Count": te_cnt,
        "Total Count": tr_cnt + va_cnt + te_cnt
    })

df_classes = pd.DataFrame(class_summary)
display(df_classes)
""")

    # Visualization
    add_md("""## 4. Multispectral Remote Sensing Visualization
Sentinel-2 multispectral bands allow composite rendering:
1. **True-Colour RGB (B4-B3-B2):** Simulates human visual perception.
2. **Color-Infrared CIR (B8-B4-B3):** Emphasizes photosynthetic biomass and vegetation density in bright red.
3. **SWIR Composite (B12-B8-B4):** Highlights moisture content, soil properties, and built-up structures.
""")
    add_code("""def stretch(img, p_low=2.0, p_high=98.0):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        ch = img[..., c]
        low, high = np.percentile(ch, (p_low, p_high))
        if high > low:
            out[..., c] = np.clip((ch - low) / (high - low), 0.0, 1.0)
        else:
            out[..., c] = np.clip(ch, 0.0, 1.0)
    return out

# Pick a sample patch
sample_idx = 0
patch = images[sample_idx]
pid = patch_ids[sample_idx]
active_lbls = [class_names[i] for i in range(len(class_names)) if labels[sample_idx, i] == 1]

# True Colour: Red=B4(idx 2), Green=B3(idx 1), Blue=B2(idx 0)
tc = stretch(np.stack([patch[2], patch[1], patch[0]], axis=-1))

# CIR: Red=B8(idx 3), Green=B4(idx 2), Blue=B3(idx 1)
cir = stretch(np.stack([patch[3], patch[2], patch[1]], axis=-1))

# SWIR: Red=B12(idx 5), Green=B8(idx 3), Blue=B4(idx 2)
swir = stretch(np.stack([patch[5], patch[3], patch[2]], axis=-1))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].imshow(tc); axes[0].set_title("True Colour (B4-B3-B2)"); axes[0].axis("off")
axes[1].imshow(cir); axes[1].set_title("Color-Infrared (B8-B4-B3)"); axes[1].axis("off")
axes[2].imshow(swir); axes[2].set_title("SWIR Composite (B12-B8-B4)"); axes[2].axis("off")
plt.suptitle(f"Patch: {pid}\\nLabels: {active_lbls}", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()
""")

    # Model definition
    add_md("""## 5. Model Architecture & Band Ablation Configurations
We define a high-efficiency **MultispectralCNN** tailored for multispectral remote sensing.
It supports:
- **M1:** 3 channels [B02, B03, B04] (Visible RGB)
- **M2:** 4 channels [B02, B03, B04, B08] (Visible + NIR)
- **M3:** 6 channels [B02, B03, B04, B08, B11, B12] (Full Multispectral)

The model produces **7 independent unconstrained logits**, optimized via `nn.BCEWithLogitsLoss()`.
""")
    add_code("""class MultispectralDataset(Dataset):
    def __init__(self, imgs, lbls, channel_indices):
        self.images = torch.from_numpy(imgs[:, channel_indices, :, :]).float()
        self.labels = torch.from_numpy(lbls).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class MultispectralCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 120x120 -> 60x60
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 60x60 -> 30x30
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 30x30 -> 15x15
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 15x15 -> 7x7
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feats = self.features(x)
        pooled = self.pool(feats)
        return self.classifier(pooled)

BAND_CONFIGS = {
    "M1": {"name": "M1_Visible_RGB", "bands": ["B02", "B03", "B04"], "indices": [0, 1, 2], "in_channels": 3},
    "M2": {"name": "M2_Visible_NIR", "bands": ["B02", "B03", "B04", "B08"], "indices": [0, 1, 2, 3], "in_channels": 4},
    "M3": {"name": "M3_Full_Multispectral", "bands": ["B02", "B03", "B04", "B08", "B11", "B12"], "indices": [0, 1, 2, 3, 4, 5], "in_channels": 6}
}
""")

    # Training function
    add_md("""## 6. Training Pipeline with Early Stopping
We use:
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** `AdamW` (learning rate = $10^{-3}$, weight decay = $10^{-4}$)
- **Scheduler:** `CosineAnnealingLR`
- **Early Stopping:** Monitored on validation loss (patience = 8 epochs)
- **Random Seed:** 42 across all runs
""")
    add_code("""def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_model(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            logits = model(imgs)
            loss = criterion(logits, tgts)
            total_loss += loss.item() * len(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(tgts.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    probs_arr = np.vstack(all_probs)
    tgts_arr = np.vstack(all_targets)
    preds_bin = (probs_arr >= threshold).astype(int)
    macro_f1 = float(f1_score(tgts_arr, preds_bin, average="macro", zero_division=0))
    return avg_loss, macro_f1, probs_arr, tgts_arr

def train_ablation_model(model_name, band_cfg, batch_size=16, max_epochs=35, patience=8, seed=42):
    set_seed(seed)
    in_channels = band_cfg["in_channels"]
    indices = band_cfg["indices"]

    train_ds = MultispectralDataset(images[train_mask], labels[train_mask], indices)
    val_ds = MultispectralDataset(images[val_mask], labels[val_mask], indices)
    test_ds = MultispectralDataset(images[test_mask], labels[test_mask], indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MultispectralCNN(in_channels=in_channels, num_classes=len(class_names)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None
    history = []

    print(f"\\n--- Training {model_name} ({band_cfg['bands']}) ---")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        for imgs, tgts in train_loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, tgts)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * len(imgs)

        train_loss = train_loss_total / len(train_loader.dataset)
        val_loss, val_macro_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_macro_f1})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), test_loader, val_loader
""")

    # Train all 3
    add_md("""## 7. Running the Three Band Ablation Experiments (M1, M2, M3)
Now we execute training for M1 (RGB), M2 (RGB+NIR), and M3 (Full Multispectral).
""")
    add_code("""trained_models = {}
histories = {}
loaders = {}

for m_key in ["M1", "M2", "M3"]:
    model, hist, test_ld, val_ld = train_ablation_model(m_key, BAND_CONFIGS[m_key])
    trained_models[m_key] = model
    histories[m_key] = hist
    loaders[m_key] = {"test": test_ld, "val": val_ld}
""")

    # Evaluation
    add_md("""## 8. Multi-Label Evaluation & Comparison
We evaluate on the independent Test set ($N=100$) and compute:
- **Macro-F1** and **Micro-F1**
- **Per-class Precision, Recall, F1, and Support**
- **False Positives (FP)** and **False Negatives (FN)**
""")
    add_code("""criterion = nn.BCEWithLogitsLoss()
comp_rows = []
per_class_rows = []

for m_key in ["M1", "M2", "M3"]:
    model = trained_models[m_key]
    test_ld = loaders[m_key]["test"]
    _, _, probs, tgts = evaluate_model(model, test_ld, criterion, device, threshold=0.5)

    preds = (probs >= 0.5).astype(int)
    macro_f1 = f1_score(tgts, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(tgts, preds, average="micro", zero_division=0)
    macro_p = precision_score(tgts, preds, average="macro", zero_division=0)
    macro_r = recall_score(tgts, preds, average="macro", zero_division=0)
    exact_match = np.mean(np.all(tgts == preds, axis=1))
    h_loss = hamming_loss(tgts, preds)

    comp_rows.append({
        "Model": m_key,
        "Macro-F1": round(macro_f1, 4),
        "Micro-F1": round(micro_f1, 4),
        "Macro-Precision": round(macro_p, 4),
        "Macro-Recall": round(macro_r, 4),
        "Exact Match": round(exact_match, 4),
        "Hamming Loss": round(h_loss, 4)
    })

    prec_c = precision_score(tgts, preds, average=None, zero_division=0)
    rec_c = recall_score(tgts, preds, average=None, zero_division=0)
    f1_c = f1_score(tgts, preds, average=None, zero_division=0)
    supp_c = tgts.sum(axis=0).astype(int)

    for i, c in enumerate(class_names):
        tp = int(np.sum((tgts[:, i] == 1) & (preds[:, i] == 1)))
        fp = int(np.sum((tgts[:, i] == 0) & (preds[:, i] == 1)))
        fn = int(np.sum((tgts[:, i] == 1) & (preds[:, i] == 0)))
        per_class_rows.append({
            "Model": m_key, "Class": c, "Precision": round(prec_c[i], 3),
            "Recall": round(rec_c[i], 3), "F1": round(f1_c[i], 3), "Support": supp_c[i],
            "TP": tp, "FP": fp, "FN": fn
        })

df_comp = pd.DataFrame(comp_rows)
df_pc = pd.DataFrame(per_class_rows)

print("=== Overall Model Comparison ===")
display(df_comp)
print("\\n=== Per-Class Performance ===")
display(df_pc)
""")

    # Visualizations
    add_md("""## 9. Comparative Performance Figures
We plot:
1. Training and Validation Loss Curves
2. Macro-F1 Comparison Bar Chart
3. Per-Class F1-Score Breakdown
""")
    add_code("""# 1. Loss curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for idx, (m, ax) in enumerate(zip(["M1", "M2", "M3"], axes)):
    h = histories[m]
    ax.plot(h["epoch"], h["train_loss"], label="Train Loss", color="#1f77b4")
    ax.plot(h["epoch"], h["val_loss"], label="Val Loss", color="#d62728", linestyle="--")
    ax.set_title(f"{m} Loss Trajectory", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
axes[0].set_ylabel("BCE Loss")
plt.tight_layout()
plt.show()

# 2. Per-class F1 comparison
pivot_f1 = df_pc.pivot(index="Class", columns="Model", values="F1")
pivot_f1.plot(kind="bar", figsize=(12, 5), colormap="viridis", edgecolor="black")
plt.title("Per-Class F1-Score Comparison: M1 (RGB) vs M2 (NIR) vs M3 (SWIR)", fontsize=13, fontweight="bold")
plt.ylabel("F1-Score")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
""")

    # Conclusion
    add_md("""## 10. Remote Sensing Insights & Summary
- **Visible bands (M1):** Strong baseline for Forest and Agriculture due to visible chlorophyll absorption/reflectance contrast.
- **NIR Addition (M2):** Band B08 captures the high reflectance plateau of healthy vegetation cellular structure, aiding distinction between mature forest canopy and lower biomass pastures/grasslands.
- **SWIR Addition (M3):** Bands B11 and B12 capture moisture gradients and canopy water absorption, aiding wetland boundary recognition and differentiating built-up impervious surfaces from bare dry soils.
- **Preliminary Results Warning:** Because this curated dataset contains 500 patches derived from Alpine Tile `T33UUP`, performance figures represent preliminary findings and reflect regional land-cover distributions.
""")

    os.makedirs("notebooks", exist_ok=True)
    out_path = "notebooks/sentinel2_multilabel_landcover.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated Colab notebook at: {out_path}")

if __name__ == "__main__":
    build_notebook()
