import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, r"c:\Users\manda\Downloads\Remote Sensing")
from src.label_mapping import BROAD_CLASSES, NUM_CLASSES, multihot_to_labels
from src.preprocessing import SELECTED_BANDS, BAND_METADATA


def generate_all_phase2_1_artifacts():
    print("[artifacts] Loading data/curated_dataset_phase2_1.npz...")
    data = np.load("data/curated_dataset_phase2_1.npz", allow_pickle=True)
    images = data["images"]          # (500, 6, 120, 120)
    labels = data["labels"]          # (500, 7)
    pids = data["patch_ids"]         # (500,)
    splits = data["splits"]          # (500,)
    countries = data["countries"]    # (500,)
    orig_labels = data["original_labels"] # (500,)
    class_names = data["class_names"]
    band_names = data["band_names"].tolist()

    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # =========================================================================
    # 1. outputs/metrics/phase2_1_patch_inventory.csv
    # =========================================================================
    print("[artifacts] Generating outputs/metrics/phase2_1_patch_inventory.csv...")
    inventory_rows = []
    for i in range(len(pids)):
        mhot = labels[i]
        broad_lbls = multihot_to_labels(mhot)
        b_means = {f"{b}_mean": float(np.mean(images[i, idx])) for idx, b in enumerate(band_names)}
        
        row = {
            "patch_id": pids[i],
            "split": splits[i],
            "country": countries[i],
            "original_corine_labels": orig_labels[i],
            "broad_labels": "; ".join(broad_lbls),
            "multihot_vector": str([int(x) for x in mhot]),
            "num_labels": int(np.sum(mhot)),
            "refl_min": float(np.min(images[i])),
            "refl_max": float(np.max(images[i])),
            **b_means
        }
        inventory_rows.append(row)

    df_inventory = pd.DataFrame(inventory_rows)
    inventory_path = "outputs/metrics/phase2_1_patch_inventory.csv"
    df_inventory.to_csv(inventory_path, index=False)
    print(f"  -> Saved {len(df_inventory)} patch records to {inventory_path}")

    # =========================================================================
    # 2. outputs/metrics/phase2_1_class_frequencies.csv
    # =========================================================================
    print("[artifacts] Generating outputs/metrics/phase2_1_class_frequencies.csv...")
    freq_rows = []
    n_train = np.sum(splits == "train")
    n_val = np.sum(splits == "validation")
    n_test = np.sum(splits == "test")
    n_total = len(pids)

    for c_idx, c_name in enumerate(class_names):
        tr_cnt = int(np.sum(labels[splits == "train", c_idx]))
        va_cnt = int(np.sum(labels[splits == "validation", c_idx]))
        te_cnt = int(np.sum(labels[splits == "test", c_idx]))
        tot_cnt = tr_cnt + va_cnt + te_cnt

        freq_rows.append({
            "class_index": c_idx,
            "class_name": c_name,
            "train_count": tr_cnt,
            "train_pct": round((tr_cnt / n_train) * 100, 2),
            "val_count": va_cnt,
            "val_pct": round((va_cnt / n_val) * 100, 2),
            "test_count": te_cnt,
            "test_pct": round((te_cnt / n_test) * 100, 2),
            "total_count": tot_cnt,
            "total_pct": round((tot_cnt / n_total) * 100, 2)
        })

    df_freq = pd.DataFrame(freq_rows)
    freq_path = "outputs/metrics/phase2_1_class_frequencies.csv"
    df_freq.to_csv(freq_path, index=False)
    print(f"  -> Saved class frequencies to {freq_path}")

    # =========================================================================
    # 3. outputs/metrics/phase2_1_band_validation.csv
    # =========================================================================
    print("[artifacts] Auditing all 500 raw GeoTIFF patches for band validation...")
    patches_dir = "data/curated_patches"
    band_audit = {
        b: {
            "native_shape": set(),
            "raw_dtypes": set(),
            "min_raw": 1_000_000,
            "max_raw": -1,
            "total_pixels": 0,
            "gt_10000": 0,
            "lt_0": 0,
            "has_nan": 0,
            "has_inf": 0
        } for b in band_names
    }

    for pid in pids:
        for b in band_names:
            tif_file = os.path.join(patches_dir, pid, f"{pid}_{b}.tif")
            with Image.open(tif_file) as img:
                arr = np.array(img)
                band_audit[b]["native_shape"].add(arr.shape)
                band_audit[b]["raw_dtypes"].add(str(arr.dtype))
                band_audit[b]["min_raw"] = min(band_audit[b]["min_raw"], int(arr.min()))
                band_audit[b]["max_raw"] = max(band_audit[b]["max_raw"], int(arr.max()))
                band_audit[b]["total_pixels"] += arr.size
                band_audit[b]["gt_10000"] += int((arr > 10000).sum())
                band_audit[b]["lt_0"] += int((arr < 0).sum())
                band_audit[b]["has_nan"] += int(np.isnan(arr).sum())
                band_audit[b]["has_inf"] += int(np.isinf(arr).sum())

    band_rows = []
    descriptions = {
        "B02": "Blue (Atmospheric & Water Penetration)",
        "B03": "Green (Vegetation Peak Reflectance)",
        "B04": "Red (Chlorophyll Absorption)",
        "B08": "Broad NIR (Vegetation Biomass / Canopy)",
        "B11": "SWIR-1 (Moisture / Soil / Snow Discrimination)",
        "B12": "SWIR-2 (Geology / Hydrocarbon / Burn Scars)"
    }
    wavelengths = {
        "B02": 490, "B03": 560, "B04": 665,
        "B08": 842, "B11": 1610, "B12": 2190
    }
    native_res = {
        "B02": 10, "B03": 10, "B04": 10,
        "B08": 10, "B11": 20, "B12": 20
    }

    for idx, b in enumerate(band_names):
        s = band_audit[b]
        b_tensor = images[:, idx, :, :]
        pct_gt = (s["gt_10000"] / s["total_pixels"]) * 100
        
        band_rows.append({
            "band_name": b,
            "description": descriptions[b],
            "central_wavelength_nm": wavelengths[b],
            "native_res_m": native_res[b],
            "native_shape": list(s["native_shape"])[0],
            "harmonized_shape": (120, 120),
            "resampling_method": "Bilinear (order=1)" if native_res[b] == 20 else "None (Native 10m)",
            "raw_dtype": list(s["raw_dtypes"])[0],
            "harmonized_dtype": "float32",
            "raw_min_dn": s["min_raw"],
            "raw_max_dn": s["max_raw"],
            "refl_min": round(float(np.min(b_tensor)), 4),
            "refl_max": round(float(np.max(b_tensor)), 4),
            "total_pixels_audited": s["total_pixels"],
            "pixels_dn_gt_10000_count": s["gt_10000"],
            "pixels_dn_gt_10000_pct": round(pct_gt, 5),
            "pixels_dn_lt_0_count": s["lt_0"],
            "non_finite_count": s["has_nan"] + s["has_inf"]
        })

    df_bands = pd.DataFrame(band_rows)
    bands_path = "outputs/metrics/phase2_1_band_validation.csv"
    df_bands.to_csv(bands_path, index=False)
    print(f"  -> Saved band validation audit to {bands_path}")

    # =========================================================================
    # 4 & 5. outputs/figures/phase2_1_true_colour.png & phase2_1_false_colour.png
    # =========================================================================
    print("[artifacts] Selecting 9 diverse representative patches across splits and rare classes...")
    # Find patches with distinct key land cover profiles
    # Pick: 3 Train, 3 Val, 3 Test
    # Ensuring Wetland, Shrub, Water, Forest, Agriculture, Grassland, Built-up are all shown
    selected_indices = []

    # 1. Train Wetland
    tr_wet = [i for i in range(len(pids)) if splits[i] == "train" and labels[i, 5] == 1.0]
    selected_indices.append(tr_wet[0])

    # 2. Train Shrub
    tr_shrub = [i for i in range(len(pids)) if splits[i] == "train" and labels[i, 6] == 1.0 and i not in selected_indices]
    selected_indices.append(tr_shrub[0])

    # 3. Train Agriculture + Built-up
    tr_ag_bu = [i for i in range(len(pids)) if splits[i] == "train" and labels[i, 1] == 1.0 and labels[i, 3] == 1.0 and i not in selected_indices]
    selected_indices.append(tr_ag_bu[0] if tr_ag_bu else [i for i in range(len(pids)) if splits[i] == "train" and i not in selected_indices][0])

    # 4. Val Wetland
    va_wet = [i for i in range(len(pids)) if splits[i] == "validation" and labels[i, 5] == 1.0]
    selected_indices.append(va_wet[0])

    # 5. Val Shrub
    va_shrub = [i for i in range(len(pids)) if splits[i] == "validation" and labels[i, 6] == 1.0 and i not in selected_indices]
    selected_indices.append(va_shrub[0])

    # 6. Val Water + Forest
    va_wat = [i for i in range(len(pids)) if splits[i] == "validation" and labels[i, 4] == 1.0 and i not in selected_indices]
    selected_indices.append(va_wat[0] if va_wat else [i for i in range(len(pids)) if splits[i] == "validation" and i not in selected_indices][0])

    # 7. Test Wetland (e.g. from Phase 1 if applicable or genuine test)
    te_wet = [i for i in range(len(pids)) if splits[i] == "test" and labels[i, 5] == 1.0]
    selected_indices.append(te_wet[0])

    # 8. Test Shrub
    te_shrub = [i for i in range(len(pids)) if splits[i] == "test" and labels[i, 6] == 1.0 and i not in selected_indices]
    selected_indices.append(te_shrub[0])

    # 9. Test Phase 1 Inspection Patch (Water + Forest)
    p1_target = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
    if p1_target in list(pids):
        selected_indices.append(list(pids).index(p1_target))
    else:
        te_rem = [i for i in range(len(pids)) if splits[i] == "test" and i not in selected_indices]
        selected_indices.append(te_rem[0])

    print(f"  Selected sample indices: {selected_indices}")

    # Helper function for 2-98% contrast stretch
    def stretch_rgb(rgb_arr):
        stretched = np.zeros_like(rgb_arr)
        for c in range(3):
            band = rgb_arr[..., c]
            p2, p98 = np.percentile(band, (2, 98))
            if p98 > p2:
                stretched[..., c] = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
            else:
                stretched[..., c] = np.clip(band, 0.0, 1.0)
        return stretched

    # True-Colour Figure (B04: Red, B03: Green, B02: Blue)
    fig_rgb, axes_rgb = plt.subplots(3, 3, figsize=(15, 16))
    fig_rgb.suptitle(
        "Phase 2.1 Curation Audit: True-Colour (RGB) Composites (B4-B3-B2)\n"
        "Sentinel-2 L2A Genuine Patches (120×120 px, 1.2×1.2 km, 10m Ground Resolution)",
        fontsize=16, fontweight="bold", y=0.98
    )

    # False-Colour Figure (B08: NIR -> Red, B04: Red -> Green, B03: Green -> Blue)
    fig_cir, axes_cir = plt.subplots(3, 3, figsize=(15, 16))
    fig_cir.suptitle(
        "Phase 2.1 Curation Audit: False-Colour NIR Composites (B8-B4-B3)\n"
        "Sentinel-2 L2A Biomass & Hydrology Discrimination (Red = Dense Vegetation, Cyan/Dark = Water/Wetland)",
        fontsize=16, fontweight="bold", y=0.98
    )

    # Band indices in tensor: B02:0, B03:1, B04:2, B08:3, B11:4, B12:5
    for grid_idx, sample_i in enumerate(selected_indices):
        r_ax = axes_rgb[grid_idx // 3, grid_idx % 3]
        c_ax = axes_cir[grid_idx // 3, grid_idx % 3]

        cube = images[sample_i]  # (6, 120, 120)
        pid = pids[sample_i]
        short_pid = "_".join(pid.split("_")[-3:])
        sp = splits[sample_i].capitalize()
        broad_lbls = multihot_to_labels(labels[sample_i])
        lbl_str = "\n".join([f"• {c}" for c in broad_lbls])

        # True-Colour: R=B04 (2), G=B03 (1), B=B02 (0)
        rgb_raw = np.stack([cube[2], cube[1], cube[0]], axis=-1)
        rgb_vis = stretch_rgb(rgb_raw)
        r_ax.imshow(rgb_vis)
        r_ax.set_title(f"[{sp}] {short_pid}\n{lbl_str}", fontsize=10, fontweight="bold", pad=8)
        r_ax.axis("off")

        # False-Colour NIR: R=B08 (3), G=B04 (2), B=B03 (1)
        cir_raw = np.stack([cube[3], cube[2], cube[1]], axis=-1)
        cir_vis = stretch_rgb(cir_raw)
        c_ax.imshow(cir_vis)
        c_ax.set_title(f"[{sp}] {short_pid}\n{lbl_str}", fontsize=10, fontweight="bold", pad=8)
        c_ax.axis("off")

    fig_rgb.tight_layout(rect=[0, 0.03, 1, 0.95])
    true_color_path = "outputs/figures/phase2_1_true_colour.png"
    fig_rgb.savefig(true_color_path, dpi=200, bbox_inches="tight")
    plt.close(fig_rgb)
    print(f"  -> Saved True-Colour composite figure to {true_color_path}")

    fig_cir.tight_layout(rect=[0, 0.03, 1, 0.95])
    false_color_path = "outputs/figures/phase2_1_false_colour.png"
    fig_cir.savefig(false_color_path, dpi=200, bbox_inches="tight")
    plt.close(fig_cir)
    print(f"  -> Saved False-Colour NIR composite figure to {false_color_path}")

    print("\n[artifacts] All Phase 2.1 metrics and figures successfully generated!")


if __name__ == "__main__":
    generate_all_phase2_1_artifacts()
