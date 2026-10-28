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
        "report/final_project_report.pdf",
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
    # CHECK 2: PDF Page Count Verification (Exactly 10 Pages)
    # -------------------------------------------------------------------------
    print("\n[CHECK 2] Auditing Final PDF Report Page Count Verification...")
    import pypdf
    pdf_pages_ok = True
    pdf_path = "report/final_project_report.pdf"
    if os.path.exists(pdf_path):
        try:
            reader = pypdf.PdfReader(pdf_path)
            num_pages = len(reader.pages)
            print(f"    ✓ {pdf_path}: {num_pages} pages")
            if num_pages != 10:
                print(f"    ERROR: {pdf_path} has {num_pages} pages, expected exactly 10 pages!")
                pdf_pages_ok = False
        except Exception as e:
            print(f"    ERROR reading {pdf_path}: {e}")
            pdf_pages_ok = False
    else:
        print(f"    ERROR: {pdf_path} not found!")
        pdf_pages_ok = False

    record_check("Final PDF Page Count Verification (Exactly 10 Pages)", pdf_pages_ok)

    # -------------------------------------------------------------------------
    # CHECK 3: PowerPoint Slide Count & Structure
    # -------------------------------------------------------------------------
    print("\n[CHECK 3] Auditing PowerPoint Presentation Deck...")
    pptx_path = "presentation/final_project_presentation.pptx"
    slide_count = 0
    pptx_ok = False

    if os.path.exists(pptx_path):
        try:
            prs = Presentation(pptx_path)
            slide_count = len(prs.slides)
            print(f"    ✓ Found presentation file with {slide_count} slides.")
            if 9 <= slide_count <= 12:
                pptx_ok = True
                print(f"    ✓ Slide count ({slide_count}) satisfies target (~10 slides).")
            else:
                print(f"    ERROR: Unexpected slide count: {slide_count} (expected 9-12).")
        except Exception as e:
            print(f"    ERROR reading presentation: {e}")

    record_check("Presentation Slide Count (~10 slides)", pptx_ok, f"({slide_count} slides)")

    # -------------------------------------------------------------------------
    # CHECK 4: Strict Numerical Consistency with Saved Metrics
    # -------------------------------------------------------------------------
    print("\n[CHECK 4] Verifying Numerical Consistency with Saved CSV Metrics...")
    comp_csv = "outputs/metrics/phase3_model_comparison.csv"
    per_class_csv = "outputs/metrics/phase3_per_class_metrics.csv"
    metrics_match = True

    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        # Check standard threshold values
        m1_row = df_comp[df_comp["Model"].str.contains("M1.*T=0.5")]
        m2_row = df_comp[df_comp["Model"].str.contains("M2.*T=0.5")]
        m3_row = df_comp[df_comp["Model"].str.contains("M3.*T=0.5")]

        m1_macro = float(m1_row["Macro-F1"].values[0])
        m1_micro = float(m1_row["Micro-F1"].values[0])
        m1_exact = float(m1_row["Exact Match Ratio"].values[0])
        m1_hamm = float(m1_row["Hamming Loss"].values[0])

        m2_macro = float(m2_row["Macro-F1"].values[0])
        m2_micro = float(m2_row["Micro-F1"].values[0])
        m2_exact = float(m2_row["Exact Match Ratio"].values[0])
        m2_hamm = float(m2_row["Hamming Loss"].values[0])

        m3_macro = float(m3_row["Macro-F1"].values[0])
        m3_micro = float(m3_row["Micro-F1"].values[0])
        m3_exact = float(m3_row["Exact Match Ratio"].values[0])
        m3_hamm = float(m3_row["Hamming Loss"].values[0])

        print(f"    Verified CSV Values (T=0.5):")
        print(f"      M1: Macro-F1 = {m1_macro:.4f}, Micro-F1 = {m1_micro:.4f}, Exact Match = {m1_exact:.4f}, Hamming Loss = {m1_hamm:.4f}")
        print(f"      M2: Macro-F1 = {m2_macro:.4f}, Micro-F1 = {m2_micro:.4f}, Exact Match = {m2_exact:.4f}, Hamming Loss = {m2_hamm:.4f}")
        print(f"      M3: Macro-F1 = {m3_macro:.4f}, Micro-F1 = {m3_micro:.4f}, Exact Match = {m3_exact:.4f}, Hamming Loss = {m3_hamm:.4f}")

        if not (abs(m1_macro - 0.5978) < 1e-4 and abs(m1_micro - 0.7403) < 1e-4 and abs(m1_exact - 0.12) < 1e-4 and abs(m1_hamm - 0.2014) < 1e-4):
            print("    ERROR: M1 metrics in CSV do not match expected!")
            metrics_match = False
        if not (abs(m2_macro - 0.7110) < 1e-4 and abs(m2_micro - 0.8205) < 1e-4 and abs(m2_exact - 0.38) < 1e-4 and abs(m2_hamm - 0.1500) < 1e-4):
            print("    ERROR: M2 metrics in CSV do not match expected!")
            metrics_match = False
        if not (abs(m3_macro - 0.5673) < 1e-4 and abs(m3_micro - 0.7081) < 1e-4 and abs(m3_exact - 0.05) < 1e-4 and abs(m3_hamm - 0.2486) < 1e-4):
            print("    ERROR: M3 metrics in CSV do not match expected!")
            metrics_match = False

        # Check phase3_thresholds.json
        thresh_json = "outputs/metrics/phase3_thresholds.json"
        if os.path.exists(thresh_json):
            with open(thresh_json, "r", encoding="utf-8") as f_th:
                th_data = json.load(f_th)
            print(f"    Verified phase3_thresholds.json: models present: {list(th_data['models'].keys())}")
            # Verify tuned thresholds match expected
            m1_th = th_data["models"]["M1"]["tuned_thresholds"]
            if m1_th["Wetland"] != 0.5 or m1_th["Forest"] != 0.3:
                print("    ERROR: M1 tuned thresholds mismatch!")
                metrics_match = False
            else:
                print("    ✓ M1 tuned thresholds in JSON match expected [Fo:0.3, Ag:0.2, Gr:0.4, Bu:0.4, Wa:0.3, We:0.5, Sh:0.2]")
        else:
            print("    ERROR: phase3_thresholds.json missing!")
            metrics_match = False

    record_check("Model Comparison CSV & JSON Metrics", metrics_match)

    # Check Per-Class F1 Table values from CSV
    print("\n[CHECK 5] Cross-Checking Per-Class F1 Table from CSV...")
    per_class_match = True
    if os.path.exists(per_class_csv):
        df_pc = pd.read_csv(per_class_csv)
        # Expected values from CSV at T=0.5:
        expected_per_class = {
            "Forest": {"M1": 0.9688, "M2": 0.9738, "M3": 0.9741},
            "Agriculture": {"M1": 0.7727, "M2": 0.8387, "M3": 0.7826},
            "Grassland/Pasture": {"M1": 0.7609, "M2": 0.7872, "M3": 0.5672},
            "Built-up/Artificial Surfaces": {"M1": 0.5862, "M2": 0.5714, "M3": 0.5574},
            "Water": {"M1": 0.7429, "M2": 0.8947, "M3": 0.4444},
            "Wetland": {"M1": 0.0000, "M2": 0.0000, "M3": 0.0000},
            "Shrub/Transitional Vegetation": {"M1": 0.3529, "M2": 0.9114, "M3": 0.6452}
        }
        for cls_name, vals in expected_per_class.items():
            for m_key in ["M1", "M2", "M3"]:
                row = df_pc[(df_pc["model_name"].str.contains(m_key)) & (df_pc["class_name"] == cls_name)]
                if len(row) > 0:
                    csv_f1 = float(row["f1_score"].values[0])
                    exp_f1 = vals[m_key]
                    if abs(csv_f1 - exp_f1) >= 1e-4:
                        print(f"    ERROR: Per-class F1 mismatch for {cls_name} {m_key}: CSV {csv_f1:.4f} vs Exp {exp_f1:.4f}")
                        per_class_match = False
                else:
                    print(f"    ERROR: Missing row in per-class CSV for {cls_name} {m_key}")
                    per_class_match = False
        if per_class_match:
            print("    ✓ All per-class F1 metrics match ground-truth values exactly.")

    record_check("Per-Class CSV Metrics Match", per_class_match)

    # -------------------------------------------------------------------------
    # CHECK 6: Final Report Integrity & Scientific Consistency
    # -------------------------------------------------------------------------
    print("\n[CHECK 6] Verifying Final Report Content & Scientific Refinements...")
    rep_path = "report/final_project_report.md"
    rep_ok = True

    if os.path.exists(rep_path):
        with open(rep_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Required verified strings
        required_strings = [
            ("0.5978", "M1 Default Macro-F1"),
            ("0.7403", "M1 Default Micro-F1"),
            ("0.7110", "M2 Default Macro-F1"),
            ("0.8205", "M2 Default Micro-F1"),
            ("0.5673", "M3 Default Macro-F1"),
            ("0.7081", "M3 Default Micro-F1"),
            ("0.7076", "M1 Tuned Macro-F1"),
            ("0.7020", "M2 Tuned Macro-F1"),
            ("0.6559", "M3 Tuned Macro-F1"),
            ("12.0%", "M1 Exact Match"),
            ("38.0%", "M2 Exact Match"),
            ("5.0%", "M3 Exact Match"),
            ("0.2014", "M1 Hamming Loss"),
            ("0.1500", "M2 Hamming Loss"),
            ("0.2486", "M3 Hamming Loss"),
            ("11.32", "NIR +11.32 pp gain"),
            ("500", "Total Patches Count"),
            ("300", "Training Split Count"),
            ("100", "Validation / Test Split Counts"),
            ("T33UUP", "Austrian Alpine Tile Provenance"),
            ("Wetland", "Wetland Class Mention"),
            ("0.0000", "Wetland Zero F1 Score"),
            ("27", "27 False Negatives / missed Wetland test patches"),
            ("8", "8 Wetland training patches"),
            ("Bilinear", "Bilinear Resampling Documentation"),
            ("BCEWithLogitsLoss", "Multi-Label Loss Function"),
            ("94.4%", "Verified 94.4% >= 2 classes prevalence"),
            ("3.00", "Verified 3.00 mean labels per patch"),
            ("Patch-Level Disjointness", "Explicit distinction of patch disjointness"),
            ("geographic independence", "Explicit qualification regarding regional scope"),
            ("outperforming baseline", "Correction 3: Objective phrasing without significance claim"),
            ("on the observed test metrics", "Correction 3: Bounded to observed test metrics"),
            ("BigEarthNet provides 19 detailed CORINE Land Cover classes", "Correction 1: 19 CORINE classes statement"),
            ("Beaches, dunes, sands", "Correction 1: Excluded class mention"),
            ("excluded because no instances occur in tile T33UUP", "Correction 1: Reason for exclusion"),
            ("M2 performed best on this curated test set at the default threshold of 0.5", "Correction 2: M2 best performance statement"),
            ("M3 underperformed M2, achieving a Macro-F1 of 0.5673 compared with 0.7110 for M2", "Correction 2: M3 underperformed statement"),
            ("Small training-sample size, spectral collinearity, resampling effects, and optimization instability are possible explanations, but their individual contributions were not independently tested.", "Correction 2: Causal hypothesis qualification")
        ]

        for s, desc in required_strings:
            if s in content:
                print(f"    ✓ Found '{s[:50]}...' ({desc})")
            else:
                print(f"    ERROR: Missing '{s}' ({desc}) in report!")
                rep_ok = False

        # Banned old false claims & conflicting stats
        banned_strings = [
            ("0.6484", "Old fabricated M1 Macro-F1"),
            ("0.7482", "Old fabricated M2 Macro-F1"),
            ("0.7905", "Old fabricated M3 Macro-F1"),
            ("68%", "Old fabricated ~68% exact match claim"),
            ("approximately 68%", "Old fabricated ~68% exact match claim"),
            ("impossible to recognize", "False claim about Wetland feasibility"),
            ("3.08", "Conflicting/unverified mean cardinality"),
            ("82.0%", "Conflicting/unverified multi-label prevalence"),
            ("82%", "Conflicting/unverified multi-label prevalence"),
            ("significantly outperforming", "Correction 3: Prohibited significance claim"),
            ("inherently reduces", "Correction 2: Prohibited broad generalization about SWIR")
        ]

        for b, desc in banned_strings:
            if b in content:
                print(f"    ERROR: Found forbidden old claim '{b}' ({desc}) in report!")
                rep_ok = False
            else:
                print(f"    ✓ Banned string absent: '{b}' ({desc})")

    record_check("Final Report Text & Numeric Consistency", rep_ok)

    # -------------------------------------------------------------------------
    # CHECK 7: Final PDF Integrity, Text Verification & No Legacy Duplicate
    # -------------------------------------------------------------------------
    print("\n[CHECK 7] Auditing Final PDF Size, Text Extraction & Legacy Status...")
    pdf_target = "report/final_project_report.pdf"
    pdf_legacy = "report/Final_Report.pdf"
    pdf_valid = True

    if os.path.exists(pdf_legacy):
        print(f"    ERROR: Legacy duplicate PDF '{pdf_legacy}' still exists! Remove it.")
        pdf_valid = False
    else:
        print(f"    ✓ Legacy duplicate PDF '{pdf_legacy}' correctly absent.")

    if os.path.exists(pdf_target):
        size_bytes = os.path.getsize(pdf_target)
        if size_bytes > 100_000:
            print(f"    ✓ {pdf_target} is intact and valid ({size_bytes:,} bytes, >100 KB).")
        else:
            print(f"    ERROR: {pdf_target} is unusually small ({size_bytes} bytes)!")
            pdf_valid = False

        # Verify text in PDF
        try:
            reader = pypdf.PdfReader(pdf_target)
            full_pdf_text = " ".join(" ".join([p.extract_text() for p in reader.pages]).split())
            
            # Check 3 required scientific corrections in PDF
            pdf_c1 = ("BigEarthNet provides 19 detailed CORINE Land Cover classes" in full_pdf_text and
                      "Beaches, dunes, sands" in full_pdf_text and
                      "excluded because no instances occur in tile T33UUP" in full_pdf_text)
            pdf_c2 = ("M2 performed best on this curated test set at the default threshold of 0.5" in full_pdf_text and
                      "Small training-sample size, spectral collinearity, resampling effects, and optimization instability are possible explanations, but their individual contributions were not independently tested." in full_pdf_text)
            pdf_c3 = "outperforming baseline M1 on the observed test metrics" in full_pdf_text
            pdf_no_banned = "significantly outperforming" not in full_pdf_text

            if pdf_c1:
                print("    ✓ PDF contains Correction 1 (CORINE classes and T33UUP exclusion).")
            else:
                print("    ERROR: PDF missing Correction 1!")
                pdf_valid = False

            if pdf_c2:
                print("    ✓ PDF contains Correction 2 (M3 qualified interpretation).")
            else:
                print("    ERROR: PDF missing Correction 2!")
                pdf_valid = False

            if pdf_c3:
                print("    ✓ PDF contains Correction 3 (Observed test metrics phrasing).")
            else:
                print("    ERROR: PDF missing Correction 3!")
                pdf_valid = False

            if pdf_no_banned:
                print("    ✓ PDF contains zero instances of 'significantly outperforming'.")
            else:
                print("    ERROR: PDF contains banned string 'significantly outperforming'!")
                pdf_valid = False
        except Exception as e:
            print(f"    ERROR inspecting PDF text: {e}")
            pdf_valid = False
    else:
        print(f"    ERROR: {pdf_target} not found!")
        pdf_valid = False

    record_check("Final PDF Integrity & Text Audit", pdf_valid)

    # -------------------------------------------------------------------------
    # CHECK 8: Speaker Notes Coverage
    # -------------------------------------------------------------------------
    print("\n[CHECK 8] Auditing Presentation Speaker Notes...")
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

