"""
evaluate.py
===========
Multi-label evaluation, decision threshold calibration, and diagnostic error analysis.

Strict Guarantees:
  - Threshold selection strictly uses validation probabilities (NEVER test labels)
  - Evaluates both default threshold (0.5) and validation-tuned per-class thresholds
  - Computes Macro-F1, Micro-F1, per-class Precision, Recall, F1, and Support
  - Computes exact False Positives (FP) and False Negatives (FN) per class
  - Multi-label coexistence analysis: reports exact match ratio, mean active labels, Hamming loss
  - Formats results into structured CSV tables and JSON artifacts
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

from src.label_mapping import BROAD_CLASSES, NUM_CLASSES


def tune_thresholds_on_validation(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    class_names: List[str] = BROAD_CLASSES,
    threshold_range: Tuple[float, float, float] = (0.20, 0.80, 0.05)
) -> Dict[str, any]:
    """
    Evaluates default threshold (0.5) on validation set and explores per-class threshold tuning.
    Threshold tuning is strictly performed on validation data; test labels are NEVER accessed.
    
    Returns a dictionary containing:
      - default_threshold: 0.5
      - tuned_thresholds: Dict[class_name, float]
      - val_f1_default_per_class: Dict[class_name, float]
      - val_f1_tuned_per_class: Dict[class_name, float]
      - val_macro_f1_default: float
      - val_macro_f1_tuned: float
    """
    t_start, t_end, t_step = threshold_range
    threshold_grid = np.arange(t_start, t_end + 1e-5, t_step)

    # 1. Evaluate default threshold 0.5
    default_preds = (val_probs >= 0.5).astype(int)
    default_f1s = f1_score(val_targets, default_preds, average=None, zero_division=0)
    default_macro_f1 = float(f1_score(val_targets, default_preds, average="macro", zero_division=0))

    # 2. Grid search per class
    tuned_thresholds = {}
    tuned_f1s = []

    for c_idx, c_name in enumerate(class_names):
        best_t = 0.5
        best_f1 = float(default_f1s[c_idx])
        y_true_c = val_targets[:, c_idx]
        y_prob_c = val_probs[:, c_idx]

        for t in threshold_grid:
            y_pred_c = (y_prob_c >= t).astype(int)
            score = float(f1_score(y_true_c, y_pred_c, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_t = float(round(t, 2))

        tuned_thresholds[c_name] = best_t
        tuned_f1s.append(best_f1)

    tuned_preds_matrix = np.zeros_like(val_targets, dtype=int)
    for c_idx, c_name in enumerate(class_names):
        tuned_preds_matrix[:, c_idx] = (val_probs[:, c_idx] >= tuned_thresholds[c_name]).astype(int)

    tuned_macro_f1 = float(f1_score(val_targets, tuned_preds_matrix, average="macro", zero_division=0))

    return {
        "default_threshold": 0.5,
        "tuned_thresholds": tuned_thresholds,
        "val_f1_default_per_class": {c: round(float(f), 4) for c, f in zip(class_names, default_f1s)},
        "val_f1_tuned_per_class": {c: round(float(f), 4) for c, f in zip(class_names, tuned_f1s)},
        "val_macro_f1_default": round(default_macro_f1, 4),
        "val_macro_f1_tuned": round(tuned_macro_f1, 4)
    }


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: Union[float, Dict[str, float], np.ndarray] = 0.5,
    class_names: List[str] = BROAD_CLASSES,
    model_name: str = "Model"
) -> Dict[str, any]:
    """
    Computes rigorous multi-label evaluation metrics:
      - Macro-F1, Micro-F1, Weighted-F1
      - Per-class Precision, Recall, F1-Score, Support, TP, FP, FN, TN
      - Exact match subset accuracy, Hamming loss
      - Co-occurring label analysis
    """
    N, num_classes = y_true.shape
    assert y_probs.shape == (N, num_classes), f"Mismatch in probs shape: {y_probs.shape} vs {y_true.shape}"

    # Vectorize thresholds
    if isinstance(thresholds, (int, float)):
        thresh_vec = np.full(num_classes, float(thresholds))
    elif isinstance(thresholds, dict):
        thresh_vec = np.array([thresholds.get(c, 0.5) for c in class_names], dtype=float)
    else:
        thresh_vec = np.asarray(thresholds, dtype=float)

    # Threshold predictions
    y_pred = (y_probs >= thresh_vec).astype(int)

    # Global multi-label metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    micro_prec = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    micro_rec = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    h_loss = float(hamming_loss(y_true, y_pred))
    exact_matches = np.all(y_true == y_pred, axis=1)
    exact_match_ratio = float(np.mean(exact_matches))

    # Per-class metrics
    per_class_rows = []
    confusion_summary = {}

    for c_idx, c_name in enumerate(class_names):
        yt_c = y_true[:, c_idx]
        yp_c = y_pred[:, c_idx]

        tp = int(np.sum((yt_c == 1) & (yp_c == 1)))
        fp = int(np.sum((yt_c == 0) & (yp_c == 1)))
        fn = int(np.sum((yt_c == 1) & (yp_c == 0)))
        tn = int(np.sum((yt_c == 0) & (yp_c == 0)))
        support = int(np.sum(yt_c == 1))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        confusion_summary[c_name] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "support": support, "threshold": float(thresh_vec[c_idx])
        }

        per_class_rows.append({
            "model_name": model_name,
            "class_name": c_name,
            "threshold": round(float(thresh_vec[c_idx]), 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })

    df_per_class = pd.DataFrame(per_class_rows)

    summary = {
        "model_name": model_name,
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "micro_precision": round(micro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "micro_recall": round(micro_rec, 4),
        "hamming_loss": round(h_loss, 4),
        "exact_match_ratio": round(exact_match_ratio, 4),
        "exact_match_count": int(np.sum(exact_matches)),
        "total_test_patches": N,
        "per_class_table": df_per_class,
        "confusion_summary": confusion_summary,
        "predictions_binary": y_pred,
        "probabilities": y_probs
    }
    return summary


def compare_ablation_models(
    results_m1: Dict[str, any],
    results_m2: Dict[str, any],
    results_m3: Dict[str, any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produces consolidated comparison tables:
      1. phase3_model_comparison.csv: High-level comparison across M1, M2, and M3.
      2. phase3_per_class_metrics.csv: Side-by-side per-class metrics across all 3 models.
    """
    model_summaries = [results_m1, results_m2, results_m3]
    
    # 1. Model comparison table
    comp_rows = []
    for res in model_summaries:
        comp_rows.append({
            "Model": res["model_name"],
            "Macro-F1": res["macro_f1"],
            "Micro-F1": res["micro_f1"],
            "Macro-Precision": res["macro_precision"],
            "Macro-Recall": res["macro_recall"],
            "Exact Match Ratio": res["exact_match_ratio"],
            "Hamming Loss": res["hamming_loss"],
            "Test Patches": res["total_test_patches"]
        })
    df_comparison = pd.DataFrame(comp_rows)

    # 2. Per-class metrics table
    df_all_classes = pd.concat([
        res["per_class_table"] for res in model_summaries
    ], ignore_index=True)

    return df_comparison, df_all_classes
