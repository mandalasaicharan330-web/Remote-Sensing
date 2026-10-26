# Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification

> **Applied Geophysics / Remote Sensing Practical Project 6**  
> Based on the **BigEarthNet v2.0 (reBEN)** benchmark dataset (TU Berlin, 2024).

---

## 🛰️ Project Overview

This repository investigates multi-label land-cover classification from Sentinel-2 multispectral imagery. In real-world satellite patches ($1.2\text{ km} \times 1.2\text{ km}$), surfaces are heterogeneous: agricultural fields border woodlands, rivers traverse settlements, and scrubland fringes pastures. Rather than forcing an image patch into an artificial single class, this project uses a multi-label deep learning framework with multi-hot target vectors $y \in \{0, 1\}^7$ over 7 simplified broad categories:

1. **Forest**
2. **Agriculture**
3. **Grassland / Pasture**
4. **Built-up / Artificial Surfaces**
5. **Water**
6. **Wetland**
7. **Shrub / Transitional Vegetation**

### Central Research Question:
> *Can visible, NIR, and SWIR Sentinel-2 observations identify multiple land-cover types within the same image patch, and which classes are difficult to recognize when they occur together?*

---

## 📁 Repository Structure

```text
Remote Sensing/
├── notebooks/
│   └── sentinel2_multilabel_landcover.ipynb  # End-to-end Colab-executable notebook
├── src/
│   ├── __init__.py
│   ├── data_access.py                       # Streaming decompressor & Zenodo metadata loader
│   ├── data_preprocessing.py                # 20m->10m resampling, normalization & label mapping
│   ├── visualization.py                     # RGB/CIR/SWIR composites, co-occurrence & spectral plots
│   ├── model.py                             # Multispectral CNN, BCE loss & threshold tuning
│   └── evaluation.py                        # Precision/Recall/F1, M1-M3 table & error analysis
├── data/
│   ├── metadata.parquet                     # Official BigEarthNet v2.0 metadata (480,038 patches)
│   └── sample_patches/                      # 5 verified genuine Sentinel-2 test patches (6 bands each)
├── outputs/
│   ├── figures/                             # Generated RGB/FCC figures and diagnostic charts
│   └── metrics/                             # Model performance logs and comparison tables
├── report/
│   └── report_draft.md                      # Comprehensive 5-7 page technical report
├── presentation/
│   └── slide_outline.md                     # 8-10 slide presentation outline
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## ⚡ Key Technical Innovation: Selective Streaming (No 59 GiB Download!)

The full BigEarthNet Sentinel-2 archive is ~59 GiB (`BigEarthNet-S2.tar.zst`). To keep this project fast, reproducible, and accessible for student compute environments:
- We implemented an **on-the-fly HTTP streaming decompressor** in `src/data_access.py` using `curl` + `zstandard` + Python `tarfile`.
- Patches are extracted directly from the front of the official Zenodo archive and decoded into memory.
- A curated sample of 100 to 1,000 genuine patches requires **only ~14 to 140 MB of network streaming** in under 1–2 minutes, saving 99.8% of download bandwidth.

---

## 🚀 Getting Started

### 1. Installation
Clone or navigate to the repository and install required packages:
```bash
pip install -r requirements.txt
```

### 2. Running in Google Colab (Recommended)
1. Upload `notebooks/sentinel2_multilabel_landcover.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Run cells sequentially from top to bottom.
3. The notebook will automatically:
   - Download the 3.44 MB `metadata.parquet`.
   - Stream a sample of genuine Sentinel-2 multispectral patches from Zenodo.
   - Display True-Colour ($B4$-$B3$-$B2$), Color-Infrared ($B8$-$B4$-$B3$), and SWIR ($B12$-$B8$-$B4$) composites.
   - Compute the label co-occurrence matrix heatmap.
   - Train and compare models **M1 (RGB)**, **M2 (RGB+NIR)**, and **M3 (RGB+NIR+SWIR)**.
   - Run threshold tuning on validation data.
   - Output per-class metrics and confusion diagnostics.

### 3. Running Modules Locally
```python
# Stream sample patches
from src.data_access import stream_extract_patches
stream_extract_patches(output_dir="data/patches", max_patches=20)

# Load metadata
from src.data_access import load_metadata_dict
meta = load_metadata_dict("data/metadata.parquet")

# Preprocess and build dataset
from src.data_preprocessing import create_curated_npz
create_curated_npz("data/patches", meta, "data/curated_subset.npz")
```

---

## 📊 Summary of Experimental Findings

| Experiment | Bands Retained | Macro-F1 | Key Observations |
| :--- | :---: | :---: | :--- |
| **M1: Visible RGB** | $[B2, B3, B4]$ | ~0.648 | Basic colour contrast; struggles on mixed vegetation. |
| **M2: RGB + NIR** | $[B2, B3, B4, B8]$ | ~0.748 | **+10.0 pp gain.** Major jump in water ($+15.6$ pp) and forest. |
| **M3: Multispectral** | $[B2, B3, B4, B8, B11, B12]$ | ~0.791 | **+4.2 pp gain.** Boosts Built-up ($+6.5$ pp) and Wetland ($+6.6$ pp). |

### Difficult Co-Occurring Classes:
- **Agriculture vs. Grassland/Pasture:** Both share herbaceous chlorophyll green peaks. SWIR soil moisture helps decouple tilled arable land from permanent root sod.
- **Forest vs. Shrub/Transitional:** Intermediate canopy density at patch boundaries creates borderline sigmoid probabilities.

---

## 📜 Citation & Credits
- **BigEarthNet v2.0 (reBEN):** K. Clasen, L. Hackel, T. Burgert, G. Sumbul, B. Demir, V. Markl, *"reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis"*, arXiv:2407.03653, 2024.
- **License:** Community Data License Agreement – Permissive (CDLA-Permissive-1.0).
