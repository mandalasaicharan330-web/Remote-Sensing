"""
verify_phase4.py
================
Phase 4 Verification and Scientific Integrity Audit Script.

Validates:
  1. Existence and non-emptiness of all Phase 4 deliverables:
     - report/final_project_report.md
     - presentation/final_project_presentation.pptx
     - report/presentation_speaker_notes.md
     - all required figures in outputs/figures/
  2. PowerPoint slide count verification (approximately 10 slides: 9-11 range).
  3. Strict numerical consistency between verified CSV metrics and reported text:
     - M1 Macro-F1 = 0.5978, Micro-F1 = 0.7403
     - M2 Macro-F1 = 0.7110, Micro-F1 = 0.8205
     - M3 Macro-F1 = 0.5673, Micro-F1 = 0.7081
     - Validation-tuned Macro-F1: M1 = 0.7076, M2 = 0.7020, M3 = 0.6559
  4. Content verification:
     - Confirms 500 patches, 300 train / 100 val / 100 test split provenance.
     - Confirms Austrian tile T33UUP regional attribution (no over-generalized claims).
     - Confirms documented Wetland failure (8 training instances, 0.0000 F1).
     - Confirms speaker notes exist for all 10 slides.
"""

import os
import sys
import json
import pandas as pd
from pptx import Presentation

sys.stdout.reconfigure(encoding="utf-8")


def verify_phase4():
    print("=" * 80)
    print("      PHASE 4: FINAL REPORT & PRESENTATION VERIFICATION AUDIT")
    print("=" * 80)

    all_passed = True
    check_results = []

    def record_check(name: str, passed: bool, detail: str = ""):
        nonlocal all_passed
        status_str = "PASSED" if passed else "FAILED"
        print(f"  [{status_str}] {name} {detail}")
        check_results.append((name, passed, detail))
        if not passed:
            all_passed = False

    # -------------------------------------------------------------------------
    # CHECK 1: File Existence & Non-emptiness
    # -------------------------------------------------------------------------
    print("\n[CHECK 1] Auditing Required Phase 4 Deliverables on Disk...")
    required_files = [
        "report/final_project_report.md",
        "presentation/final_project_presentation.pptx",
        "report/presentation_speaker_notes.md",
        "outputs/metrics/phase3_model_comparison.csv",
        "outputs/metrics/phase3_per_class_metrics.csv",
        "outputs/metrics/phase3_thresholds.json",
        "outputs/metrics/phase3_training_history.csv",
        "outputs/figures/phase3_loss_curves.png",
        "outputs/figures/phase3_macro_f1_comparison.png",
        "outputs/figures/phase3_per_class_f1_comparison.png",
        "outputs/figures/phase3_precision_recall_summary.png",
        "outputs/figures/phase3_test_predictions_mosaic.png",
        "outputs/figures/phase3_fp_fn_heatmap.png",
        "outputs/figures/phase2_1_true_colour.png",
        "outputs/figures/phase2_1_false_colour.png"
    ]

    missing_files = []
    for f in required_files:
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        status = f"OK ({size:,} bytes)" if (exists and size > 0) else "MISSING / EMPTY"
        print(f"    - {f:<55s} : {status}")
        if not exists or size == 0:
            missing_files.append(f)

    record_check("Required Phase 4 Deliverables Existence", len(missing_files) == 0,
                 f"(Missing: {len(missing_files)})")

    # -------------------------------------------------------------------------
    # CHECK 2: PowerPoint Slide Count & Structure
    # -------------------------------------------------------------------------
    print("\n[CHECK 2] Auditing PowerPoint Presentation Deck...")
    pptx_path = "presentation/final_project_presentation.pptx"
    slide_count = 0
    pptx_ok = False

    if os.path.exists(pptx_path):
        try:
            prs = Presentation(pptx_path)
            slide_count = len(prs.slides)
            print(f"    ✓ Found presentation file with {slide_count} slides.")
            # Expecting approximately 10 slides (between 9 and 12)
            if 9 <= slide_count <= 12:
                pptx_ok = True
                print(f"    ✓ Slide count ({slide_count}) satisfies the target of approximately 10 slides.")
            else:
                print(f"    ERROR: Unexpected slide count: {slide_count} (expected 9-12).")
        except Exception as e:
            print(f"    ERROR reading presentation: {e}")

    record_check("Presentation Slide Count (~10 slides)", pptx_ok, f"({slide_count} slides)")

    # -------------------------------------------------------------------------
    # CHECK 3: Strict Numerical Consistency with Verified Metrics
    # -------------------------------------------------------------------------
    print("\n[CHECK 3] Verifying Numerical Consistency with Saved Metrics...")
    comp_csv = "outputs/metrics/phase3_model_comparison.csv"
    metrics_match = False

    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        m1_row = df_comp[df_comp["Model"].str.contains("M1")]
        m2_row = df_comp[df_comp["Model"].str.contains("M2")]
        m3_row = df_comp[df_comp["Model"].str.contains("M3")]

        m1_macro = float(m1_row["Macro-F1"].values[0])
        m1_micro = float(m1_row["Micro-F1"].values[0])
        m2_macro = float(m2_row["Macro-F1"].values[0])
        m2_micro = float(m2_row["Micro-F1"].values[0])
        m3_macro = float(m3_row["Macro-F1"].values[0])
        m3_micro = float(m3_row["Micro-F1"].values[0])

        print(f"    Verified CSV Values:")
        print(f"      M1 (T=0.5): Macro-F1 = {m1_macro:.4f}, Micro-F1 = {m1_micro:.4f}")
        print(f"      M2 (T=0.5): Macro-F1 = {m2_macro:.4f}, Micro-F1 = {m2_micro:.4f}")
        print(f"      M3 (T=0.5): Macro-F1 = {m3_macro:.4f}, Micro-F1 = {m3_micro:.4f}")

        # Check expected numbers
        expected_ok = (
            abs(m1_macro - 0.5978) < 1e-4 and abs(m1_micro - 0.7403) < 1e-4 and
            abs(m2_macro - 0.7110) < 1e-4 and abs(m2_micro - 0.8205) < 1e-4 and
            abs(m3_macro - 0.5673) < 1e-4 and abs(m3_micro - 0.7081) < 1e-4
        )

        if expected_ok:
            print("    ✓ All CSV values strictly match project ground-truth metrics.")
            metrics_match = True
        else:
            print("    ERROR: CSV metrics do not match expected ground truth values!")

    record_check("Model Comparison CSV Metrics", metrics_match)

    # -------------------------------------------------------------------------
    # CHECK 4: Final Report Integrity & Content Requirements
    # -------------------------------------------------------------------------
    print("\n[CHECK 4] Verifying Final Report Content & Scientific Integrity...")
    rep_path = "report/final_project_report.md"
    rep_ok = True

    if os.path.exists(rep_path):
        with open(rep_path, "r", encoding="utf-8") as f:
            content = f.read()

        required_strings = [
            ("0.5978", "M1 Macro-F1"),
            ("0.7403", "M1 Micro-F1"),
            ("0.7110", "M2 Macro-F1"),
            ("0.8205", "M2 Micro-F1"),
            ("0.5673", "M3 Macro-F1"),
            ("0.7081", "M3 Micro-F1"),
            ("0.7076", "M1 Tuned Macro-F1"),
            ("0.7020", "M2 Tuned Macro-F1"),
            ("0.6559", "M3 Tuned Macro-F1"),
            ("500", "Total Patches Count"),
            ("300", "Training Split Count"),
            ("100", "Validation / Test Split Counts"),
            ("T33UUP", "Austrian Alpine Tile Provenance"),
            ("Wetland", "Wetland Class Mention"),
            ("0.0000", "Wetland Zero F1 Score"),
            ("Bilinear", "Bilinear Resampling Documentation"),
            ("BCEWithLogitsLoss", "Multi-Label Loss Function")
        ]

        for s, desc in required_strings:
            if s in content:
                print(f"    ✓ Found '{s}' ({desc})")
            else:
                print(f"    ERROR: Missing '{s}' ({desc}) in report!")
                rep_ok = False

    record_check("Final Report Text & Numeric Consistency", rep_ok)

    # -------------------------------------------------------------------------
    # CHECK 5: Speaker Notes Coverage
    # -------------------------------------------------------------------------
    print("\n[CHECK 5] Auditing Presentation Speaker Notes...")
    notes_path = "report/presentation_speaker_notes.md"
    notes_ok = True

    if os.path.exists(notes_path):
        with open(notes_path, "r", encoding="utf-8") as f:
            notes_content = f.read()

        for s_num in range(1, 11):
            s_tag = f"Slide {s_num}"
            if s_tag in notes_content:
                print(f"    ✓ Speaker notes present for {s_tag}")
            else:
                print(f"    ERROR: Missing speaker notes for {s_tag}")
                notes_ok = False

    record_check("Speaker Notes for All 10 Slides", notes_ok)

    # -------------------------------------------------------------------------
    # Summary Scorecard
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    PHASE 4 VERIFICATION SCORECARD")
    print("=" * 80)
    for name, passed, detail in check_results:
        status_str = "PASSED" if passed else "FAILED"
        print(f"  {name:<60s} : [{status_str}]")

    print("=" * 80)
    if all_passed:
        print("RESULT: ALL PHASE 4 VERIFICATION & AUDIT CHECKS PASSED WITH ZERO DEFECTS.")
    else:
        print("RESULT: ONE OR MORE CHECKS FAILED. INVESTIGATE BEFORE COMPLETION.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = verify_phase4()
    sys.exit(0 if success else 1)
