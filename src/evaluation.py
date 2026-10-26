"""
evaluation.py
=============
Multi-label evaluation metrics, comparison tables, and diagnostic error analysis.
- Computes per-class Precision, Recall, and F1.
- Computes overall Macro-F1.
- Builds comparison tables for M1 (RGB), M2 (RGB+NIR), and M3 (RGB+NIR+SWIR).
- Diagnoses missed classes, false extra classes, and confusing class pairs
  (Agriculture vs Grassland, Forest vs Transitional Vegetation).
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> pd.DataFrame:
    """
    Computes per-class Precision, Recall, F1-score, and support counts,
    along with overall Macro and Micro averages.
    """
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = np.sum(y_true, axis=0).astype(int)

    df = pd.DataFrame({
        "Class": class_names,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Support": support
    })

    # Macro average row
    macro_row = pd.DataFrame([{
        "Class": "Macro Average",
        "Precision": np.mean(prec),
        "Recall": np.mean(rec),
        "F1-Score": np.mean(f1),
        "Support": int(np.sum(support))
    }])

    return pd.concat([df, macro_row], ignore_index=True)


def build_m1_m2_m3_comparison(
    metrics_m1: pd.DataFrame,
    metrics_m2: pd.DataFrame,
    metrics_m3: pd.DataFrame,
    class_names: List[str]
) -> pd.DataFrame:
    """
    Generates a consolidated comparison table comparing F1 scores across
    M1 (RGB), M2 (RGB+NIR), and M3 (RGB+NIR+SWIR).
    """
    classes_list = list(class_names) + ["Macro Average"]
    rows = []

    for c in classes_list:
        f1_m1 = metrics_m1.loc[metrics_m1["Class"] == c, "F1-Score"].values[0]
        f1_m2 = metrics_m2.loc[metrics_m2["Class"] == c, "F1-Score"].values[0]
        f1_m3 = metrics_m3.loc[metrics_m3["Class"] == c, "F1-Score"].values[0]
        gain_nir = (f1_m2 - f1_m1) * 100
        gain_swir = (f1_m3 - f1_m2) * 100
        total_gain = (f1_m3 - f1_m1) * 100

        rows.append({
            "Class": c,
            "M1 (RGB) F1": round(f1_m1, 4),
            "M2 (RGB+NIR) F1": round(f1_m2, 4),
            "M3 (All Bands) F1": round(f1_m3, 4),
            "NIR Gain (pp)": round(gain_nir, 2),
            "SWIR Gain (pp)": round(gain_swir, 2),
            "Total Gain (pp)": round(total_gain, 2)
        })

    return pd.DataFrame(rows)


def analyze_prediction_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, any]:
    """
    Categorizes test predictions into:
    - Exactly correct patches (all labels match)
    - Patches with missed labels (False Negatives)
    - Patches with false extra labels (False Positives)
    - Specific pairwise confusion between Agriculture vs Grassland
      and Forest vs Transitional Vegetation.
    """
    exact_matches = np.all(y_true == y_pred, axis=1)
    missed_any = np.any((y_true == 1) & (y_pred == 0), axis=1)
    extra_any = np.any((y_true == 0) & (y_pred == 1), axis=1)

    c_map = {name: i for i, name in enumerate(class_names)}

    # Agriculture vs Grassland confusion
    ag_idx = c_map.get("Agriculture")
    gr_idx = c_map.get("Grassland_Pasture")
    ag_gr_confusion = 0
    if ag_idx is not None and gr_idx is not None:
        # True is Agriculture, predicted is Grassland (or vice versa) without the other
        c1 = (y_true[:, ag_idx] == 1) & (y_pred[:, gr_idx] == 1) & (y_true[:, gr_idx] == 0)
        c2 = (y_true[:, gr_idx] == 1) & (y_pred[:, ag_idx] == 1) & (y_true[:, ag_idx] == 0)
        ag_gr_confusion = int(np.sum(c1 | c2))

    # Forest vs Shrub/Transitional confusion
    fo_idx = c_map.get("Forest")
    sh_idx = c_map.get("Shrub_Transitional")
    fo_sh_confusion = 0
    if fo_idx is not None and sh_idx is not None:
        c3 = (y_true[:, fo_idx] == 1) & (y_pred[:, sh_idx] == 1) & (y_true[:, sh_idx] == 0)
        c4 = (y_true[:, sh_idx] == 1) & (y_pred[:, fo_idx] == 1) & (y_true[:, fo_idx] == 0)
        fo_sh_confusion = int(np.sum(c3 | c4))

    return {
        "total_test_samples": len(y_true),
        "exact_match_count": int(np.sum(exact_matches)),
        "exact_match_ratio": float(np.mean(exact_matches)),
        "patches_with_missed_labels": int(np.sum(missed_any)),
        "patches_with_false_extra_labels": int(np.sum(extra_any)),
        "agriculture_grassland_confusion_count": ag_gr_confusion,
        "forest_shrub_confusion_count": fo_sh_confusion
    }
