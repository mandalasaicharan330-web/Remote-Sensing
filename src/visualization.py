"""
visualization.py
================
Visual exploration and remote sensing diagnostics for Sentinel-2 multispectral imagery.
- True-Colour (B4-B3-B2) and False-Colour (CIR: B8-B4-B3, SWIR: B12-B8-B4) composites.
- Label frequency and distribution of labels per patch.
- Label co-occurrence matrix heatmap.
- Mean spectral signatures across visible, NIR, and SWIR bands.
- Prediction error diagnostics (missed classes, false alarms).
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.label_mapping import BROAD_CLASSES

BAND_LABELS = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)", "B11 (SWIR1)", "B12 (SWIR2)"]
BAND_WAVELENGTHS = [490, 560, 665, 842, 1610, 2190]  # center wavelengths in nm


def stretch_composite(rgb_stack: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    """
    Applies percentile contrast stretching across RGB channels for display.
    Expects rgb_stack of shape (H, W, 3).
    """
    out = np.zeros_like(rgb_stack, dtype=np.float32)
    for c in range(3):
        ch = rgb_stack[..., c]
        p_low, p_high = np.percentile(ch, (lower_pct, upper_pct))
        if p_high > p_low:
            stretched = np.clip((ch - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            stretched = np.clip(ch, 0.0, 1.0)
        out[..., c] = stretched
    return out


def make_true_colour(patch_tensor: np.ndarray) -> np.ndarray:
    """
    Constructs a true-colour composite: Red = B04, Green = B03, Blue = B02.
    patch_tensor shape: (6, H, W) where order is [B02, B03, B04, B08, B11, B12].
    """
    b2 = patch_tensor[0]
    b3 = patch_tensor[1]
    b4 = patch_tensor[2]
    rgb = np.stack([b4, b3, b2], axis=-1)
    return stretch_composite(rgb)


def make_false_colour_cir(patch_tensor: np.ndarray) -> np.ndarray:
    """
    Constructs a standard Color-Infrared (CIR) composite:
    Red = B08 (NIR), Green = B04 (Red), Blue = B03 (Green).
    Highlights dense vegetation in bright red, bare soil in tan, water in black.
    """
    b3 = patch_tensor[1]
    b4 = patch_tensor[2]
    b8 = patch_tensor[3]
    rgb = np.stack([b8, b4, b3], axis=-1)
    return stretch_composite(rgb)


def make_false_colour_swir(patch_tensor: np.ndarray) -> np.ndarray:
    """
    Constructs a SWIR-NIR-Red composite:
    Red = B12 (SWIR-2), Green = B08 (NIR), Blue = B04 (Red).
    Highlights moisture gradients, built-up surfaces, and vegetation types.
    """
    b4 = patch_tensor[2]
    b8 = patch_tensor[3]
    b12 = patch_tensor[5]
    rgb = np.stack([b12, b8, b4], axis=-1)
    return stretch_composite(rgb)


def plot_patch_multispectral_views(
    patch_tensor: np.ndarray,
    patch_id: str,
    labels: List[str],
    save_path: Optional[str] = None
):
    """
    Plots True-Colour, CIR, and SWIR composites side by side for a patch.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    tc = make_true_colour(patch_tensor)
    cir = make_false_colour_cir(patch_tensor)
    swir = make_false_colour_swir(patch_tensor)

    axes[0].imshow(tc)
    axes[0].set_title("True Colour (B4-B3-B2)\nVisible Spectrum", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(cir)
    axes[1].set_title("Color-Infrared (B8-B4-B3)\nBiomass & Vegetation Structure", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(swir)
    axes[2].set_title("SWIR Composite (B12-B8-B4)\nMoisture, Soil & Built-Up", fontsize=11)
    axes[2].axis("off")

    lbl_str = ", ".join(labels)
    plt.suptitle(f"Patch: {patch_id}\nLabels: [{lbl_str}]", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualization] Saved multi-spectral composite to {save_path}")
    plt.show()


def plot_class_frequencies(
    labels_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plots the frequency of each land-cover class in the curated dataset.
    """
    counts = np.sum(labels_matrix, axis=0)
    sorted_idx = np.argsort(counts)[::-1]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        [class_names[i] for i in sorted_idx],
        [counts[i] for i in sorted_idx],
        color="#2b5c8f",
        edgecolor="#16324f"
    )
    plt.title("Class Frequency Distribution in Curated Dataset", fontsize=13, fontweight="bold")
    plt.xlabel("Land-Cover Category", fontsize=11)
    plt.ylabel("Number of Patches", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2, f"{int(h)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_labels_per_patch(
    labels_matrix: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plots the distribution of number of active labels per patch.
    """
    labels_count = np.sum(labels_matrix, axis=1).astype(int)
    unique, counts = np.unique(labels_count, return_counts=True)

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(unique, counts, color="#348e71", edgecolor="#1b4d3e", width=0.6)
    plt.title("Distribution of Co-Occurring Labels per Image Patch", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Labels Assigned to a Single Patch", fontsize=10)
    plt.ylabel("Number of Patches", fontsize=10)
    plt.xticks(unique)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(labels_count)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., h + 1, f"{int(h)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def compute_and_plot_cooccurrence(
    labels_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Computes co-occurrence matrix C_ij = number of patches containing both class i and j,
    and visualizes it as an annotated heatmap.
    """
    # C = Y.T @ Y
    co_matrix = np.dot(labels_matrix.T, labels_matrix).astype(int)

    plt.figure(figsize=(9, 7.5))
    im = plt.imshow(co_matrix, cmap="YlGnBu", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.title("Label Co-Occurrence Matrix ($C_{ij}$)", fontsize=13, fontweight="bold")
    plt.xticks(range(len(class_names)), class_names, rotation=35, ha="right", fontsize=10)
    plt.yticks(range(len(class_names)), class_names, fontsize=10)

    # Annotate numbers
    thresh = co_matrix.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = co_matrix[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return co_matrix


def plot_mean_spectral_signatures(
    images_matrix: np.ndarray,
    labels_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Computes mean spectral response s_k = [B2, B3, B4, B8, B11, B12] for each class
    and plots signature curves against Sentinel-2 band center wavelengths.
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for k, cname in enumerate(class_names):
        # Select all patches containing class k
        mask = labels_matrix[:, k] == 1.0
        if not np.any(mask):
            continue
        # Mean across patches and spatial dimensions (H, W): shape (6,)
        mean_sig = np.mean(images_matrix[mask], axis=(0, 2, 3))
        plt.plot(
            range(6),
            mean_sig,
            marker="o",
            linewidth=2,
            label=f"{cname} (n={np.sum(mask)})",
            color=colors[k]
        )

    plt.xticks(range(6), [f"{b}\n({w} nm)" for b, w in zip(BAND_LABELS, BAND_WAVELENGTHS)], fontsize=9)
    plt.ylabel("Mean Surface Reflectance", fontsize=11)
    plt.title("Mean Multispectral Signatures ($\bar{s}_k$) for Land-Cover Classes", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ablation_loss_curves(
    history_df: "pd.DataFrame",
    save_path: Optional[str] = "outputs/figures/phase3_loss_curves.png"
):
    """
    Plots training and validation loss curves across epochs for M1, M2, and M3 models.
    """
    models = history_df["model_name"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    colors = {"train": "#1f77b4", "val": "#d62728"}

    for ax, m in zip(axes, models):
        m_df = history_df[history_df["model_name"] == m]
        ax.plot(m_df["epoch"], m_df["train_loss"], label="Train Loss (BCE)", color=colors["train"], linewidth=2)
        ax.plot(m_df["epoch"], m_df["val_loss"], label="Val Loss (BCE)", color=colors["val"], linewidth=2, linestyle="--")

        # Mark best epoch
        best_row = m_df.loc[m_df["val_loss"].idxmin()]
        ax.scatter([best_row["epoch"]], [best_row["val_loss"]], color="#2ca02c", s=80, zorder=5,
                   label=f"Best Val ({best_row['val_loss']:.3f})")

        ax.set_title(f"{m}\n(Best Epoch {int(best_row['epoch'])})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

    axes[0].set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    plt.suptitle("M1 / M2 / M3 Training and Validation Loss Trajectories", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualization] Saved loss curves to {save_path}")
    plt.close()


def plot_macro_f1_comparison(
    comparison_df: "pd.DataFrame",
    save_path: Optional[str] = "outputs/figures/phase3_macro_f1_comparison.png"
):
    """
    Bar chart comparing Macro-F1, Micro-F1, and Exact Match Ratio across M1, M2, and M3.
    """
    models = comparison_df["Model"].tolist()
    macro_f1 = comparison_df["Macro-F1"].tolist()
    micro_f1 = comparison_df["Micro-F1"].tolist()
    exact_match = comparison_df["Exact Match Ratio"].tolist()

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(x - width, macro_f1, width, label="Macro-F1", color="#2b5c8f", edgecolor="#16324f")
    rects2 = ax.bar(x, micro_f1, width, label="Micro-F1", color="#2ca02c", edgecolor="#1b631b")
    rects3 = ax.bar(x + width, exact_match, width, label="Exact Match Ratio", color="#e377c2", edgecolor="#a24886")

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_title("Overall Multi-Label Performance Comparison: M1 vs M2 vs M3", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Attach labels on bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualization] Saved macro F1 comparison to {save_path}")
    plt.close()


def plot_per_class_f1_comparison(
    per_class_df: "pd.DataFrame",
    save_path: Optional[str] = "outputs/figures/phase3_per_class_f1_comparison.png"
):
    """
    Grouped bar chart comparing Per-Class F1-Score across M1, M2, and M3.
    """
    classes = per_class_df["class_name"].unique()
    models = per_class_df["model_name"].unique()

    x = np.arange(len(classes))
    total_width = 0.75
    single_width = total_width / len(models)
    palette = ["#4a7bb0", "#f28e2b", "#59a14f"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (m, color) in enumerate(zip(models, palette)):
        m_data = per_class_df[per_class_df["model_name"] == m].set_index("class_name")
        f1_vals = [m_data.loc[c, "f1_score"] if c in m_data.index else 0.0 for c in classes]
        offset = (idx - (len(models) - 1) / 2) * single_width
        rects = ax.bar(x + offset, f1_vals, single_width, label=m, color=color, edgecolor="#333333", alpha=0.9)

        for rect in rects:
            h = rect.get_height()
            if h > 0.01:
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Per-Class F1-Score Comparison Across Band Configurations (M1 / M2 / M3)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualization] Saved per-class F1 comparison to {save_path}")
    plt.close()


def plot_precision_recall_summary(
    per_class_df: "pd.DataFrame",
    save_path: Optional[str] = "outputs/figures/phase3_precision_recall_summary.png"
):
    """
    Plots Precision vs Recall per class for each model.
    """
    models = per_class_df["model_name"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True, sharex=True)
    if len(models) == 1:
        axes = [axes]

    classes = per_class_df["class_name"].unique()
    color_map = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for ax, m in zip(axes, models):
        m_df = per_class_df[per_class_df["model_name"] == m].set_index("class_name")
        for idx, c in enumerate(classes):
            if c in m_df.index:
                p = m_df.loc[c, "precision"]
                r = m_df.loc[c, "recall"]
                supp = m_df.loc[c, "support"]
                ax.scatter(r, p, s=max(40, supp * 4), color=color_map[idx], label=c, alpha=0.85, edgecolors="black")

        # Iso-F1 curves
        f_scores = [0.2, 0.4, 0.6, 0.8]
        for f in f_scores:
            x_vals = np.linspace(0.01, 1, 100)
            y_vals = (f * x_vals) / (2 * x_vals - f + 1e-9)
            mask = (y_vals >= 0) & (y_vals <= 1)
            ax.plot(x_vals[mask], y_vals[mask], color="gray", alpha=0.3, linestyle=":")
            ax.text(0.92, (f * 0.92) / (2 * 0.92 - f + 1e-9), f"F1={f}", color="gray", fontsize=7)

        ax.set_title(f"{m} (Precision vs Recall)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Precision", fontsize=10)
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, fontsize=8)
    plt.suptitle("Class-wise Precision-Recall Diagnostics across Ablation Models", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualization] Saved PR summary to {save_path}")
    plt.close()


def plot_fp_fn_heatmap(
    per_class_df: "pd.DataFrame",
    save_path: Optional[str] = "outputs/figures/phase3_fp_fn_heatmap.png"
):
    """
    Heatmap comparing False Positives (FP) and False Negatives (FN) per class across models.
    """
    classes = per_class_df["class_name"].unique()
    models = per_class_df["model_name"].unique()

    # Form FP and FN matrices
    fp_mat = np.zeros((len(classes), len(models)))
    fn_mat = np.zeros((len(classes), len(models)))

    for j, m in enumerate(models):
        m_df = per_class_df[per_class_df["model_name"] == m].set_index("class_name")
        for i, c in enumerate(classes):
            if c in m_df.index:
                fp_mat[i, j] = m_df.loc[c, "fp"]
                fn_mat[i, j] = m_df.loc[c, "fn"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # FP Heatmap
    im1 = axes[0].imshow(fp_mat, cmap="YlOrRd", aspect="auto")
    axes[0].set_title("False Positives (FP - Commission Errors)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, fontsize=10, fontweight="bold")
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_yticklabels(classes, fontsize=9)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(len(classes)):
        for j in range(len(models)):
            val = int(fp_mat[i, j])
            color = "white" if val > fp_mat.max() / 2.0 else "black"
            axes[0].text(j, i, f"{val}", ha="center", va="center", color=color, fontweight="bold")

    # FN Heatmap
    im2 = axes[1].imshow(fn_mat, cmap="PuBuGn", aspect="auto")
    axes[1].set_title("False Negatives (FN - Omission Errors)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, fontsize=10, fontweight="bold")
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_yticklabels(classes, fontsize=9)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    for i in range(len(classes)):
        for j in range(len(models)):
            val = int(fn_mat[i, j])
            color = "white" if val > fn_mat.max() / 2.0 else "black"
            axes[1].text(j, i, f"{val}", ha="center", va="center", color=color, fontweight="bold")

    plt.suptitle("Class-wise Error Diagnosis: False Positives vs False Negatives", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualization] Saved FP/FN heatmap to {save_path}")
    plt.close()


def plot_test_predictions_mosaic(
    test_images: np.ndarray,
    test_patch_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str] = BROAD_CLASSES,
    sample_indices: Optional[List[int]] = None,
    save_path: Optional[str] = "outputs/figures/phase3_test_predictions_mosaic.png"
):
    """
    Visualizes selected test patches with True vs Predicted multi-label classifications.
    Uses True-Colour composite for display.
    """
    if sample_indices is None:
        # Pick 6 diverse patches (e.g. including wetland or shrub, and different classes)
        sample_indices = [0, 5, 12, 25, 45, 60]
        # Filter to valid bounds
        sample_indices = [idx for idx in sample_indices if idx < len(test_images)]

    num_samples = len(sample_indices)
    cols = 3
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5.5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for plot_idx, s_idx in enumerate(sample_indices):
        ax = axes_flat[plot_idx]
        patch_tensor = test_images[s_idx]
        pid = str(test_patch_ids[s_idx])

        # Composite: Red=B04 (index 2), Green=B03 (index 1), Blue=B02 (index 0)
        tc_img = make_true_colour(patch_tensor)
        ax.imshow(tc_img)
        ax.axis("off")

        # True labels
        true_classes = [class_names[i] for i in range(len(class_names)) if y_true[s_idx, i] == 1]
        pred_classes = [f"{class_names[i]} ({y_probs[s_idx, i]:.2f})"
                        for i in range(len(class_names)) if y_pred[s_idx, i] == 1]

        t_str = "\n".join(true_classes) if true_classes else "[None]"
        p_str = "\n".join(pred_classes) if pred_classes else "[None]"

        # Exact match indicator
        is_exact = np.array_equal(y_true[s_idx], y_pred[s_idx])
        match_color = "green" if is_exact else "crimson"
        match_symbol = "✓ Exact Match" if is_exact else "✗ Discrepancy"

        title_text = f"Patch: ...{pid[-16:]}\n" \
                     f"Ground Truth:\n{t_str}\n" \
                     f"Predicted:\n{p_str}\n" \
                     f"[{match_symbol}]"

        ax.set_title(title_text, fontsize=9, loc="left", color=match_color, fontweight="bold")

    # Hide unused subplots
    for remaining in range(num_samples, len(axes_flat)):
        axes_flat[remaining].axis("off")

    plt.suptitle("Sample Test Patches: Ground Truth vs Predicted Multi-Labels", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[visualization] Saved test prediction mosaic to {save_path}")
    plt.close()

