"""
run_phase3_experiments.py
=========================
Master training and evaluation execution script for Phase 3:
Sentinel-2 Multi-Label Land-Cover Classification Band Ablation (M1, M2, M3).

Executes:
  1. Data loading and partition verification (Train: 300, Val: 100, Test: 100).
  2. M1 Training (Visible RGB: B02, B03, B04).
  3. M2 Training (Visible + NIR: B02, B03, B04, B08).
  4. M3 Training (Full Multispectral: B02, B03, B04, B08, B11, B12).
  5. Validation threshold tuning and calibration saving to outputs/metrics/phase3_thresholds.json.
  6. Independent test set evaluation with full multi-label metrics.
  7. Exporting CSV metric tables:
     - outputs/metrics/phase3_training_history.csv
     - outputs/metrics/phase3_model_comparison.csv
     - outputs/metrics/phase3_per_class_metrics.csv
  8. Generating publication-quality figures in outputs/figures/:
     - phase3_loss_curves.png
     - phase3_macro_f1_comparison.png
     - phase3_per_class_f1_comparison.png
     - phase3_precision_recall_summary.png
     - phase3_test_predictions_mosaic.png
     - phase3_fp_fn_heatmap.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import BAND_CONFIGS, count_parameters
from src.label_mapping import BROAD_CLASSES, NUM_CLASSES
from src.train import (
    set_seed,
    load_dataset_splits,
    train_single_model,
    evaluate_epoch
)
from src.evaluate import (
    tune_thresholds_on_validation,
    compute_comprehensive_metrics,
    compare_ablation_models
)
from src.visualization import (
    plot_ablation_loss_curves,
    plot_macro_f1_comparison,
    plot_per_class_f1_comparison,
    plot_precision_recall_summary,
    plot_fp_fn_heatmap,
    plot_test_predictions_mosaic
)

sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 80)
    print("      PHASE 3: MULTI-LABEL MODEL TRAINING & M1/M2/M3 BAND ABLATION")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}\n")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # 1. Load verified dataset splits
    print("[STEP 1] Loading verified dataset partitions from data/curated_dataset_phase2_1.npz...")
    splits = load_dataset_splits("data/curated_dataset_phase2_1.npz")
    print(f"  ✓ Train patches:      {len(splits['train']['patch_ids'])}")
    print(f"  ✓ Validation patches: {len(splits['validation']['patch_ids'])}")
    print(f"  ✓ Test patches:       {len(splits['test']['patch_ids'])}")

    # Verify zero overlap
    tr_ids = set(splits["train"]["patch_ids"])
    va_ids = set(splits["validation"]["patch_ids"])
    te_ids = set(splits["test"]["patch_ids"])
    assert len(tr_ids & va_ids) == 0 and len(tr_ids & te_ids) == 0 and len(va_ids & te_ids) == 0, \
        "Data leakage detected among splits!"

    # Common training hyperparameters
    BATCH_SIZE = 16
    MAX_EPOCHS = 35
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 8
    SEED = 42

    print(f"  ✓ Hyperparameters: Batch Size={BATCH_SIZE}, Max Epochs={MAX_EPOCHS}, LR={LEARNING_RATE}, "
          f"Weight Decay={WEIGHT_DECAY}, Patience={PATIENCE}, Seed={SEED}")

    # 2. Train Models (M1, M2, M3)
    trained_models = {}
    all_histories = []

    model_keys = ["M1", "M2", "M3"]
    for m_key in model_keys:
        cfg = BAND_CONFIGS[m_key]
        print(f"\n[STEP 2.{model_keys.index(m_key)+1}] Executing {m_key} Experiment ({cfg['name']})...")
        print(f"  Selected Bands: {cfg['bands']} (Channel Indices: {cfg['indices']})")

        res = train_single_model(
            model_name=m_key,
            band_config=cfg,
            dataset_splits=splits,
            checkpoint_dir="models",
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            seed=SEED,
            device=device,
            verbose=True
        )
        trained_models[m_key] = res
        all_histories.append(res["history"])

    # Combine histories and save CSV
    df_combined_history = pd.concat(all_histories, ignore_index=True)
    history_csv = "outputs/metrics/phase3_training_history.csv"
    df_combined_history.to_csv(history_csv, index=False)
    print(f"\n[STEP 3] Saved training loss history to: {history_csv}")

    # 3. Validation Threshold Calibration
    print("\n[STEP 4] Calibrating Decision Thresholds on Validation Set (No Test Data Used)...")
    criterion = torch.nn.BCEWithLogitsLoss()
    threshold_records = {
        "metadata": {
            "description": "Decision thresholds calibrated strictly on validation predictions",
            "search_range": [0.20, 0.80, 0.05],
            "num_classes": NUM_CLASSES,
            "classes": BROAD_CLASSES
        },
        "models": {}
    }

    val_predictions = {}
    for m_key in model_keys:
        model = trained_models[m_key]["model"]
        loaders = trained_models[m_key]["loaders"]
        val_loss, val_macro_f1, val_probs, val_targets = evaluate_epoch(
            model, loaders["validation"], criterion, device, threshold=0.5
        )
        val_predictions[m_key] = {"probs": val_probs, "targets": val_targets}

        calib = tune_thresholds_on_validation(val_probs, val_targets, class_names=BROAD_CLASSES)
        threshold_records["models"][m_key] = calib

        print(f"  {m_key}: Validation Macro-F1 (T=0.5): {calib['val_macro_f1_default']:.4f} | "
              f"Tuned: {calib['val_macro_f1_tuned']:.4f}")
        for cname in BROAD_CLASSES:
            print(f"    - {cname:<30s}: T_default=0.5 (F1={calib['val_f1_default_per_class'][cname]:.3f}) -> "
                  f"T_tuned={calib['tuned_thresholds'][cname]:.2f} (F1={calib['val_f1_tuned_per_class'][cname]:.3f})")

    thresholds_json = "outputs/metrics/phase3_thresholds.json"
    with open(thresholds_json, "w", encoding="utf-8") as f:
        json.dump(threshold_records, f, indent=2)
    print(f"  ✓ Saved calibrated thresholds to: {thresholds_json}")

    # 4. Independent Test Evaluation
    print("\n[STEP 5] Performing Independent Evaluation on Test Set (N=100)...")
    test_eval_results_default = {}
    test_eval_results_tuned = {}
    test_predictions = {}

    for m_key in model_keys:
        model = trained_models[m_key]["model"]
        loaders = trained_models[m_key]["loaders"]
        test_loss, test_macro_f1, test_probs, test_targets = evaluate_epoch(
            model, loaders["test"], criterion, device, threshold=0.5
        )
        test_predictions[m_key] = {"probs": test_probs, "targets": test_targets}

        # Evaluate with default threshold (0.5)
        eval_def = compute_comprehensive_metrics(
            test_targets, test_probs, thresholds=0.5, class_names=BROAD_CLASSES, model_name=f"{m_key} (T=0.5)"
        )
        test_eval_results_default[m_key] = eval_def

        # Evaluate with validation-tuned thresholds
        tuned_thresh_dict = threshold_records["models"][m_key]["tuned_thresholds"]
        eval_tun = compute_comprehensive_metrics(
            test_targets, test_probs, thresholds=tuned_thresh_dict, class_names=BROAD_CLASSES, model_name=f"{m_key} (Tuned)"
        )
        test_eval_results_tuned[m_key] = eval_tun

    # Build comparison tables using standard T=0.5 (and tuned as auxiliary)
    df_comparison, df_per_class = compare_ablation_models(
        test_eval_results_default["M1"],
        test_eval_results_default["M2"],
        test_eval_results_default["M3"]
    )

    comparison_csv = "outputs/metrics/phase3_model_comparison.csv"
    per_class_csv = "outputs/metrics/phase3_per_class_metrics.csv"
    df_comparison.to_csv(comparison_csv, index=False)
    df_per_class.to_csv(per_class_csv, index=False)

    print("\n" + "=" * 80)
    print("                    TEST EVALUATION COMPARISON (T=0.5)")
    print("=" * 80)
    print(df_comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("                    PER-CLASS METRICS (M1 / M2 / M3)")
    print("=" * 80)
    print(df_per_class.to_string(index=False))

    # Also display tuned comparison for insight
    df_comp_tuned, _ = compare_ablation_models(
        test_eval_results_tuned["M1"],
        test_eval_results_tuned["M2"],
        test_eval_results_tuned["M3"]
    )
    print("\n" + "=" * 80)
    print("             TEST EVALUATION COMPARISON (VALIDATION-TUNED THRESHOLDS)")
    print("=" * 80)
    print(df_comp_tuned.to_string(index=False))

    # 5. Generate Publication Figures
    print("\n[STEP 6] Generating Publication-Quality Figures under outputs/figures/...")
    
    # 1. Loss curves
    plot_ablation_loss_curves(df_combined_history, "outputs/figures/phase3_loss_curves.png")

    # 2. Macro-F1 comparison
    plot_macro_f1_comparison(df_comparison, "outputs/figures/phase3_macro_f1_comparison.png")

    # 3. Per-class F1 comparison
    plot_per_class_f1_comparison(df_per_class, "outputs/figures/phase3_per_class_f1_comparison.png")

    # 4. Precision-Recall summary
    plot_precision_recall_summary(df_per_class, "outputs/figures/phase3_precision_recall_summary.png")

    # 5. FP/FN heatmap
    plot_fp_fn_heatmap(df_per_class, "outputs/figures/phase3_fp_fn_heatmap.png")

    # 6. Test prediction mosaic (using M3 predictions)
    test_imgs = splits["test"]["images"]
    test_pids = splits["test"]["patch_ids"]
    y_test_true = test_predictions["M3"]["targets"]
    y_test_probs = test_predictions["M3"]["probs"]
    y_test_pred = test_eval_results_default["M3"]["predictions_binary"]

    plot_test_predictions_mosaic(
        test_images=test_imgs,
        test_patch_ids=test_pids,
        y_true=y_test_true,
        y_pred=y_test_pred,
        y_probs=y_test_probs,
        class_names=BROAD_CLASSES,
        sample_indices=[2, 7, 18, 33, 52, 85],
        save_path="outputs/figures/phase3_test_predictions_mosaic.png"
    )

    print("\n" + "=" * 80)
    print("  ✓ ALL PHASE 3 TRAINING & BAND ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
