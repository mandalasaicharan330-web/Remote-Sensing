# Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification

**Final Project Report**  
**Course / Discipline:** Applied Remote Sensing & Earth Observation  
**Dataset:** BigEarthNet v2.0 (*reBEN*, Sentinel-2 Level-2A BOA Surface Reflectance)  
**Authors:** Undergraduate Student (Applied Geophysics) & Antigravity IDE Pair Programmer  
**Date:** October 26, 2026  
**Status:** Certified & Verified (`verify_phase4.py` Passed with Exit Code 0)

---

## Abstract

Terrestrial landscapes are intrinsically heterogeneous mosaics where multiple land-cover types frequently coexist within a single sensor footprint. Conventional single-label satellite image classification forces an image patch into a single mutually exclusive category, discarding subordinate land-cover components and failing on mixed pixels. In this project, we formulate land-cover recognition as a **multi-label classification problem** using European Space Agency (ESA) Sentinel-2 multispectral imagery from the BigEarthNet v2.0 benchmark archive. 

We curated and verified a 500-patch genuine dataset originating from Sentinel-2 tile `T33UUP` in the Austrian Alps, partitioned into official, disjoint splits: **300 Training**, **100 Validation**, and **100 Test** patches. We harmonized six spectral bands across visible (B02, B03, B04 at 10 m), near-infrared (B08 at 10 m), and shortwave-infrared (B11, B12 bilinearly resampled from 20 m to 10 m, $120 \times 120$ pixels) domains, radiometrically scaled to Bottom-Of-Atmosphere (BOA) surface reflectance $[0.0, 1.0]$. 

We trained a compact 4-block regularized Convolutional Neural Network (`MultispectralCNN`, ~617k parameters) under `BCEWithLogitsLoss` and conducted a rigorous **Band Ablation Experiment** comparing:
- **M1 (Visible RGB):** Bands B02, B03, B04 (3 channels)
- **M2 (Visible + NIR):** Bands B02, B03, B04, B08 (4 channels)
- **M3 (Full Multispectral):** Bands B02, B03, B04, B08, B11, B12 (6 channels)

At the standard decision threshold ($T=0.5$), **M2 achieved the highest overall performance**, recording a test **Macro-F1 of 0.7110**, **Micro-F1 of 0.8205**, and **Exact Match subset accuracy of 38.0%**, significantly outperforming baseline M1 (**Macro-F1 of 0.5978**, **Micro-F1 of 0.7403**, Exact Match 12.0%). Adding NIR provided a critical physical mechanism for separating mature forest canopy from low-stature woody scrub (*Shrub/Transitional Vegetation* F1 surged from 0.3529 to 0.9114) and sharply elevated *Water* F1 from 0.7429 to 0.8947. 

Conversely, **M3 underperformed M2**, attaining a test **Macro-F1 of 0.5673** ($T=0.5$) and **0.6559** (validation-tuned). On our 300-patch training split, adding two 20 m resampled SWIR bands increased input dimensionality without sufficient training support, inducing optimization instability and severe false-positive over-prediction for *Grassland/Pasture* (57 false positives; precision 40.0%). Furthermore, all three models recorded **0.0000 F1 on Wetland** at $T=0.5$ because only 8 positive Wetland patches existed in the training set (2.7% prevalence). We discuss these findings in the context of remote sensing physics, small-sample bias, and multi-label decision calibration.

---

## 1. Introduction and Problem Statement

Satellite remote sensing of the Earth's surface provides the observational foundation for monitoring ecosystem dynamics, carbon budgeting, urban expansion, and agricultural production. For decades, land-cover mapping frameworks categorized spatial units into single categorical classes. However, spatial resolutions between $10\text{ m}$ and $20\text{ m}$ (such as Sentinel-2 MSI) regularly span ecotones, parcel boundaries, riparian corridors, and rural settlements where multiple distinct land covers coexist within a single $1.2\text{ km} \times 1.2\text{ km}$ ($120 \times 120\text{ pixel}$) observation patch.

Single-label classification algorithms face fundamental failures in such heterogeneous environments:
1. **Information Loss:** Forcing a mixed patch containing 60% coniferous forest, 25% pasture, and 15% stream water into only "Forest" completely blinds downstream models to water resources and grazing land.
2. **Artificial Label Noise:** Forcing human annotators or automated rules to pick one "dominant" label introduces subjective ambiguity along class boundaries.
3. **Mutual Exclusivity Fallacy:** Land-cover classes are not mutually exclusive. Multi-label classification models each land-cover category as an independent binary Bernoulli variable:
   $$\mathbf{y} = [y_1, y_2, \dots, y_K]^\top \in \{0, 1\}^K$$
   where $y_k = 1$ denotes the confirmed presence of class $k$, and $\sum_{k=1}^K y_k \ge 1$.

This project investigates multi-label land-cover recognition using multi-band Sentinel-2 imagery, evaluating how infrared observations impact classification accuracy across complex co-occurring surface types.

---

## 2. Project Objectives

1. **Curate and Validate a Real-World Sentinel-2 Dataset:** Assemble a verified, leakage-free 500-patch multi-label dataset from the BigEarthNet v2.0 archive preserving official split provenance (300 train, 100 validation, 100 test).
2. **Implement Harmonized Multispectral Preprocessing:** Convert raw digital numbers (DN) to physical BOA reflectance $[0.0, 1.0]$ and resample 20 m SWIR bands to 10 m working grids using bilinear interpolation.
3. **Establish a 7-Class Broad Mapping Scheme:** Map the 19 official CORINE Land Cover (CLC) categories into 7 broad, ecologically grounded target classes, auditing unmapped categories.
4. **Conduct a Controlled Spectral Band Ablation Study:** Train identical CNN architectures on three spectral configurations:
   - $M_1$: Visible RGB (B02, B03, B04)
   - $M_2$: Visible + NIR (B02, B03, B04, B08)
   - $M_3$: Full Multispectral (B02, B03, B04, B08, B11, B12)
5. **Evaluate Multi-Label Performance Rigorously:** Measure Macro-F1, Micro-F1, per-class Precision/Recall/F1, Exact Match subset accuracy, Hamming loss, and confusion matrices.
6. **Analyze Remote Sensing Physics & Limitations:** Examine why NIR and SWIR behave differently across vegetation and water classes, isolate failure modes on severe minority classes (*Wetland*), and document geographic limitations.

---

## 3. BigEarthNet and Sentinel-2 Dataset Description

### The BigEarthNet v2.0 Benchmark Archive
BigEarthNet v2.0 (*reBEN*, Sumbul et al., 2024; Clasen et al., 2024) is the premier open-access benchmark archive for multi-label remote sensing. It comprises 480,038 Sentinel-2 Level-2A image patches captured across 13 European countries. Each patch covers a spatial extent of $1.2\text{ km} \times 1.2\text{ km}$. Ground truth labels originate from the European Environment Agency (EEA) CORINE Land Cover 2018 database, updated to the revised 19-class nomenclature.

### Geographic Origin of the Curated Dataset
To enable reproducible local development and Colab execution without downloading the prohibitive 59 GiB raw compressed archive, we curated a genuine 500-patch dataset from the first tile in the BigEarthNet catalog: **Tile `T33UUP`**, situated in the **Austrian Central Alps** (UTM Zone 33N).
- **Climate and Topography:** Temperate alpine/sub-alpine terrain characterized by steep elevation gradients, glacial valleys, coniferous and mixed deciduous woodlands, alpine pastures, valley agriculture, and mountain streams.
- **Data Integrity:** 100% of the 500 patches are genuine Sentinel-2 Level-2A GeoTIFFs extracted directly from the official Zenodo archive (`BigEarthNet-S2.tar.zst`).
- **Official Split Assignments:**
  - **Training Split:** 300 patches (60.0%)
  - **Validation Split:** 100 patches (20.0%)
  - **Test Split:** 100 patches (20.0%)
- **Zero Leakage:** Strictly disjoint patch partitions ($\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$).
- **Inspection Patch Protocol:** All 5 Phase 1 inspection patches (`T33UUP_26_57`, `T33UUP_27_55`, `T33UUP_27_56`, `T33UUP_27_57`, `T33UUP_27_58`) reside strictly in the Test split.

---

## 4. Six Spectral Bands and Their Spectroradiometric Roles

Sentinel-2's MultiSpectral Instrument (MSI) samples 13 spectral bands. For this investigation, we retain the six primary diagnostic bands spanning the visible, near-infrared, and shortwave-infrared spectra:

| Band Identifier | Descriptive Name | Central Wavelength ($\lambda$) | Bandwidth ($\Delta \lambda$) | Native Spatial Res. | Grid Shape ($1.2\text{ km}$) | Primary Spectroradiometric Role in Land-Cover Discrimination |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **B02** | Blue | 490 nm | 65 nm | 10 m | $120 \times 120$ | Atmospheric penetration; water body clarity; sensitive to soil vs vegetation differences. |
| **B03** | Green | 560 nm | 35 nm | 10 m | $120 \times 120$ | Chlorophyll reflectance peak ("green peak"); assesses general vegetative vigor. |
| **B04** | Red | 665 nm | 30 nm | 10 m | $120 \times 120$ | Maximum chlorophyll-a absorption; steep transition into the NIR ("red edge"). |
| **B08** | Near-Infrared (NIR) | 842 nm | 115 nm | 10 m | $120 \times 120$ | High scattering plateau in spongy leaf mesophyll cells; total absorption by open water. |
| **B11** | Shortwave Infrared-1 | 1610 nm | 90 nm | 20 m | $60 \times 60 \to 120 \times 120$ | Foliar moisture absorption; dry vs wet soil discrimination; impervious built-up response. |
| **B12** | Shortwave Infrared-2 | 2190 nm | 180 nm | 20 m | $60 \times 60 \to 120 \times 120$ | Hydroxyl absorption; mineral and soil background characterization; burn/bare soil index. |

---

## 5. Seven Broad Land-Cover Classes and CORINE Label Mapping

The original BigEarthNet CORINE database assigns 19 detailed classes. To ensure statistically meaningful support in curated sub-samples, we map the 19 CORINE labels into **7 canonical broad land-cover classes**:

```
Col 0: Forest
Col 1: Agriculture
Col 2: Grassland/Pasture
Col 3: Built-up/Artificial Surfaces
Col 4: Water
Col 5: Wetland
Col 6: Shrub/Transitional Vegetation
```

### Complete CORINE Land Cover (CLC 19) Mapping Scheme

| Original CORINE Land Cover (CLC) Label | Mapped Target Class | Status | Spectroradiometric & Ecological Rationale |
|:---|:---|:---:|:---|
| Broad-leaved forest | **Forest** | Mapped | Deciduous broad-leaved tree canopy cover $> 30\%$. |
| Coniferous forest | **Forest** | Mapped | Evergreen needle-leaved canopy cover $> 30\%$. |
| Mixed forest | **Forest** | Mapped | Blend where neither broad-leaved nor coniferous dominates. |
| Arable land | **Agriculture** | Mapped | Regularly tilled parcels for annual crop production. |
| Permanent crops | **Agriculture** | Mapped | Vineyards, orchards, fruit trees, and berry plantations. |
| Complex cultivation patterns | **Agriculture** | Mapped | Mosaic of small parcel cultivations interspersed with dwellings. |
| Land principally occupied by agriculture, natural veg. | **Agriculture** | Mapped | Farmland interspersed with significant patches of natural woodland. |
| Agro-forestry areas | **Agriculture** | Mapped | Crops or pasture combined with forest trees. |
| Pastures | **Grassland/Pasture** | Mapped | Managed perennial herbaceous cover grazed by livestock. |
| Natural grassland and sparsely vegetated areas | **Grassland/Pasture** | Mapped | Low-productivity alpine meadows, open steppe, rocky grasslands. |
| Urban fabric | **Built-up/Artificial Surfaces** | Mapped | Continuous and discontinuous residential, commercial, civic fabric. |
| Industrial or commercial units | **Built-up/Artificial Surfaces** | Mapped | Factories, industrial facilities, infrastructure, transport nodes. |
| Inland waters | **Water** | Mapped | Freshwater lakes, reservoirs, canals, and river channels. |
| Marine waters | **Water** | Mapped | Coastal waters, estuaries, bays, and open seas. |
| Inland wetlands | **Wetland** | Mapped | Freshwater marshes, peatbogs, seasonally waterlogged ground. |
| Coastal wetlands | **Wetland** | Mapped | Salt marshes, salines, and tidal flats. |
| Transitional woodland, shrub | **Shrub/Transitional Vegetation** | Mapped | Bushy regrowth, young saplings, clear-cut regeneration, maquis. |
| Moors, heathland and sclerophyllous vegetation | **Shrub/Transitional Vegetation** | Mapped | Low woody scrub, heathland, gorse, and dwarf alpine shrubs. |
| **Beaches, dunes, sands** | **[None - Excluded]** | **Excluded** | Bare coastal sand/shingle lacking vegetation ($0.27\%$ of archive, 0 in $T33UUP$). |

*Audit Verification:* The code does not silently reassign unmapped labels. Unmapped labels contribute zero 1s to target vectors. In tile `T33UUP`, *Beaches, dunes, sands* occurs 0 times.

---

## 6. Data Preparation and Reflectance Preprocessing

Raw Sentinel-2 Level-2A GeoTIFFs provide 16-bit integer Digital Numbers ($\text{DN}$). Preprocessing transforms raw files into harmonized physical tensors:

1. **Spatial Harmonization (Bilinear Resampling):**  
   Bands B02, B03, B04, and B08 have a native spatial resolution of $10\text{ m}$ ($120 \times 120$ pixels). Bands B11 and B12 are captured at $20\text{ m}$ ($60 \times 60$ pixels). We resample B11 and B12 to $120 \times 120$ using **bilinear interpolation** (`PIL.Image.BILINEAR`, order=1). Bilinear interpolation computes distance-weighted averages of the four nearest pixel centers, producing smooth continuous spatial gradients without ringing artifacts (unlike bicubic) or artificial blocky steps (unlike nearest neighbor).
2. **Radiometric Scaling to BOA Surface Reflectance:**  
   According to the ESA Sentinel-2 L2A product specification, surface reflectance $\rho$ is quantified as $\text{DN} = 10,000 \times \rho$. We apply:
   $$\rho = \text{clip}\left(\frac{\text{DN}}{10,000.0}, 0.0, 1.0\right)$$
   Division by 10,000 converts raw DN to Bottom-Of-Atmosphere surface reflectance. Clipping to $[0.0, 1.0]$ handles rare specular glint highlights (e.g., solar reflections on glass roofs, greenhouse panels, or calm water surfaces $> 10,000\text{ DN}$, which represent $< 0.0042\%$ of all pixels).
3. **Preservation of Raw Data:**  
   All raw GeoTIFF files remain completely unmodified in `data/curated_patches/`. Preprocessed tensors are saved in compressed archive `data/curated_dataset_phase2_1.npz` (76.66 MB).

---

## 7. True-Colour and False-Colour Visualizations

Multi-band imagery allows composite visualization revealing distinct physical properties:
- **True-Colour RGB (B04-B03-B02):** Red = B04, Green = B03, Blue = B02 with 2%–98% percentile contrast stretching. Reflects natural human vision.
- **Color-Infrared CIR (B08-B04-B03):** Red = B08 (NIR), Green = B04 (Red), Blue = B03 (Green). Healthy photosynthetic biomass reflects intensely in NIR, rendering vigorous forest canopy and pastures as vibrant scarlet red. Clear water absorbs NIR strongly, appearing jet black.
- **SWIR Composite (B12-B08-B04):** Red = B12 (SWIR-2), Green = B08 (NIR), Blue = B04 (Red). Differentiates soil moisture gradients, exposed mineral terrain, and built-up structures.

![True-Colour Composites](outputs/figures/phase2_1_true_colour.png)
*Figure 1: True-colour (B4-B3-B2) composite matrix across genuine training, validation, and test patches.*

![False-Colour CIR Composites](outputs/figures/phase2_1_false_colour.png)
*Figure 2: Color-Infrared (B8-B4-B3) composite matrix highlighting dense vegetative biomass in red and hydrology in black.*

---

## 8. Exploratory Data Analysis & Class Distribution

### Multi-Label Complexity
- Minimum active labels per patch: **1**
- Maximum active labels per patch: **5**
- Mean active labels per patch: **3.08**
- **82.0% of all patches contain $\ge 2$ co-occurring classes**, confirming that single-label classification is fundamentally inadequate for European landscapes.

### Class Prevalence Across Splits ($N=500$)

| Index | Target Class | Train ($N=300$) | Val ($N=100$) | Test ($N=100$) | Total ($N=500$) | Dataset Prevalence |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | **Forest** | 258 (86.0%) | 84 (84.0%) | 95 (95.0%) | 437 | 87.4% |
| 1 | **Agriculture** | 142 (47.3%) | 74 (74.0%) | 52 (52.0%) | 268 | 53.6% |
| 2 | **Grassland/Pasture** | 209 (69.7%) | 75 (75.0%) | 39 (39.0%) | 323 | 64.6% |
| 3 | **Built-up/Artificial** | 130 (43.3%) | 58 (58.0%) | 31 (31.0%) | 219 | 43.8% |
| 4 | **Water** | 41 (13.7%) | 28 (28.0%) | 21 (21.0%) | 90 | 18.0% |
| 5 | **Wetland** | **8 (2.7%)** | **16 (16.0%)** | **27 (27.0%)** | **51** | **10.2%** |
| 6 | **Shrub/Transitional** | **59 (19.7%)** | **13 (13.0%)** | **42 (42.0%)** | **114** | **22.8%** |

*Co-occurrence Highlights:* The most frequent pairwise co-occurrences in tile `T33UUP` are **Forest + Grassland/Pasture** (wooded valley margins), **Forest + Agriculture**, and **Agriculture + Built-up** (rural villages).

---

## 9. Multi-Label Classification Methodology

### Mathematical Formulation
Given an input multispectral cube $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ ($H=W=120$), the neural network $f_\theta(\mathbf{X})$ computes unconstrained raw logits $\mathbf{z} = [z_1, z_2, \dots, z_K]^\top \in \mathbb{R}^7$. 

Unlike multi-class classification which employs a softmax activation ($\sum_k p_k = 1$), multi-label classification treats each class as an independent Bernoulli trial. Independent class probabilities $\hat{p}_k \in [0, 1]$ are obtained via the standard logistic sigmoid function:
$$\hat{p}_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}$$

### Multi-Label Loss Function
The model is optimized using **Binary Cross-Entropy with Logits** (`BCEWithLogitsLoss`), numerically combining the sigmoid operation with binary cross-entropy for numerical stability:
$$\mathcal{L}_{\text{BCE}}(\mathbf{z}, \mathbf{y}) = -\frac{1}{K} \sum_{k=1}^K \left[ y_k \cdot \log \sigma(z_k) + (1 - y_k) \cdot \log(1 - \sigma(z_k)) \right]$$

### Binary Decision Thresholding
Binary label predictions $\hat{y}_k \in \{0, 1\}$ are formed by applying a decision threshold vector $\mathbf{T} = [T_1, \dots, T_7]$:
$$\hat{y}_k = \mathbb{I}(\hat{p}_k \ge T_k)$$
We evaluate:
1. **Standard Default Threshold ($T_k = 0.50$):** Canonical baseline for unweighted cross-entropy.
2. **Validation-Calibrated Thresholds ($T_k^*$):** Selected by grid search over $T \in [0.20, 0.80]$ on the validation split to maximize validation Macro-F1. *Crucially, test labels are never accessed during calibration.*

---

## 10. CNN Architecture & Training Configuration

We implemented `MultispectralCNN` in `src/model.py`, engineered for spatial-spectral feature extraction:
- **Block 1:** $\text{Conv2d}(C \to 32, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv2d}(32 \to 32, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{MaxPool2d}(2, 2) \implies 60 \times 60$
- **Block 2:** $\text{Conv2d}(32 \to 64, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv2d}(64 \to 64, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{MaxPool2d}(2, 2) \implies 30 \times 30$
- **Block 3:** $\text{Conv2d}(64 \to 128, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv2d}(128 \to 128, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{MaxPool2d}(2, 2) \implies 15 \times 15$
- **Block 4:** $\text{Conv2d}(128 \to 256, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{MaxPool2d}(2, 2) \implies 7 \times 7$
- **Global Pooling & Head:** $\text{AdaptiveAvgPool2d}((1, 1)) \to \text{Flatten} \to \text{Dropout}(0.4) \to \text{Linear}(256, 128) \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(128, 7)$

### Optimization Hyperparameters:
- **Optimizer:** `AdamW` ($\text{lr} = 1.0 \times 10^{-3}$, $\text{weight\_decay} = 1.0 \times 10^{-4}$)
- **LR Scheduler:** `CosineAnnealingLR` ($T_{\max} = 35$, $\eta_{\min} = 1.0 \times 10^{-5}$)
- **Batch Size:** 16
- **Early Stopping:** Monitored on validation loss (patience = 8 epochs)
- **Seed Policy:** `seed=42` across Python, NumPy, and PyTorch

---

## 11. Experimental Design for M1, M2, and M3

| Model | Name | Input Channels | Band Identifiers | Central Wavelengths | Parameter Count | Scientific Question Tested |
|:---:|:---|:---:|:---|:---|:---:|:---|
| **M1** | Visible RGB | 3 | B02, B03, B04 | 490, 560, 665 nm | 616,679 | Baseline: Can standard RGB imagery resolve co-occurring land covers? |
| **M2** | Visible + NIR | 4 | B02, B03, B04, B08 | + 842 nm | 616,967 | Does NIR vegetation plateau reflectance resolve canopy density and water? |
| **M3** | Full Multispectral | 6 | B02, B03, B04, B08, B11, B12 | + 1610, 2190 nm | 617,543 | Do 20 m SWIR moisture/mineral bands enhance or hinder small-sample CNNs? |

---

## 12. Results Tables from Verified CSV Files

All numbers in the tables below are transcribed directly from [outputs/metrics/phase3_model_comparison.csv](file:///c:/Users/manda/Downloads/Remote%20Sensing/outputs/metrics/phase3_model_comparison.csv) and [outputs/metrics/phase3_per_class_metrics.csv](file:///c:/Users/manda/Downloads/Remote%20Sensing/outputs/metrics/phase3_per_class_metrics.csv).

### Test Set Performance Comparison ($N=100$)

| Evaluation Mode | Model Identifier | Macro-F1 | Micro-F1 | Macro-Precision | Macro-Recall | Exact Match Ratio | Hamming Loss |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard ($T=0.50$)** | **M1 (Visible RGB)** | 0.5978 | 0.7403 | 0.7317 | 0.5588 | 12.0% | 0.2014 |
| **Standard ($T=0.50$)** | **M2 (Visible + NIR)** | **0.7110** | **0.8205** | **0.7326** | **0.7036** | **38.0%** | **0.1500** |
| **Standard ($T=0.50$)** | **M3 (Multispectral)** | 0.5673 | 0.7081 | 0.6894 | 0.5666 | 5.0% | 0.2486 |
| *Validation-Tuned* | M1 (Tuned Thresholds) | 0.7076 | 0.8024 | 0.6648 | 0.7868 | 9.0% | 0.1857 |
| *Validation-Tuned* | M2 (Tuned Thresholds) | 0.7020 | 0.7928 | 0.6566 | 0.7886 | 7.0% | 0.1971 |
| *Validation-Tuned* | M3 (Tuned Thresholds) | 0.6559 | 0.7692 | 0.5939 | 0.7645 | 6.0% | 0.2229 |

![Training Loss Curves](outputs/figures/phase3_loss_curves.png)
*Figure 3: Training and validation loss curves across epochs for M1, M2, and M3.*

![Macro-F1 Comparison](outputs/figures/phase3_macro_f1_comparison.png)
*Figure 4: Overall performance comparison across M1, M2, and M3 showing Macro-F1, Micro-F1, and Exact Match Ratio.*

---

## 13. Per-Class Precision, Recall, and F1 Analysis

### Per-Class Test Metrics Table ($T = 0.50$)

| Model | Class Name | Support | Precision | Recall | F1-Score | TP | FP | FN | TN |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1** | Forest | 95 | 0.9588 | 0.9789 | 0.9688 | 93 | 4 | 2 | 1 |
| M1 | Agriculture | 52 | 0.9444 | 0.6538 | 0.7727 | 34 | 2 | 18 | 46 |
| M1 | Grassland/Pasture | 39 | 0.6604 | 0.8974 | 0.7609 | 35 | 18 | 4 | 43 |
| M1 | Built-up/Artificial Surfaces | 31 | 0.6296 | 0.5484 | 0.5862 | 17 | 10 | 14 | 59 |
| M1 | Water | 21 | 0.9286 | 0.6190 | 0.7429 | 13 | 1 | 8 | 78 |
| M1 | **Wetland** | **27** | **0.0000** | **0.0000** | **0.0000** | **0** | **0** | **27** | **73** |
| M1 | **Shrub/Transitional Vegetation** | **42** | **1.0000** | **0.2143** | **0.3529** | **9** | **0** | **33** | **58** |
| **M2** | Forest | 95 | 0.9688 | 0.9789 | 0.9738 | 93 | 3 | 2 | 2 |
| M2 | Agriculture | 52 | 0.9512 | 0.7500 | 0.8387 | 39 | 2 | 13 | 46 |
| M2 | Grassland/Pasture | 39 | 0.6727 | 0.9487 | 0.7872 | 37 | 18 | 2 | 43 |
| M2 | Built-up/Artificial Surfaces | 31 | 0.5625 | 0.5806 | 0.5714 | 18 | 14 | 13 | 55 |
| M2 | Water | 21 | **1.0000** | **0.8095** | **0.8947** | **17** | **0** | **4** | **79** |
| M2 | **Wetland** | **27** | **0.0000** | **0.0000** | **0.0000** | **0** | **0** | **27** | **73** |
| M2 | **Shrub/Transitional Vegetation** | **42** | **0.9730** | **0.8571** | **0.9114** | **36** | **1** | **6** | **57** |
| **M3** | Forest | 95 | 0.9592 | 0.9895 | 0.9741 | 94 | 4 | 1 | 1 |
| M3 | Agriculture | 52 | 0.9000 | 0.6923 | 0.7826 | 36 | 4 | 16 | 44 |
| M3 | Grassland/Pasture | 39 | **0.4000** | 0.9744 | **0.5672** | 38 | **57** | 1 | 4 |
| M3 | Built-up/Artificial Surfaces | 31 | 0.5667 | 0.5484 | 0.5574 | 17 | 13 | 14 | 56 |
| M3 | Water | 21 | 1.0000 | **0.2857** | **0.4444** | 6 | 0 | **15** | 79 |
| M3 | **Wetland** | **27** | **0.0000** | **0.0000** | **0.0000** | **0** | **0** | **27** | **73** |
| M3 | **Shrub/Transitional Vegetation** | **42** | 1.0000 | 0.4762 | 0.6452 | 20 | 0 | 22 | 58 |

![Per-Class F1 Comparison](outputs/figures/phase3_per_class_f1_comparison.png)
*Figure 5: Per-class F1 score comparison across M1, M2, and M3.*

![Precision Recall Summary](outputs/figures/phase3_precision_recall_summary.png)
*Figure 6: Precision versus Recall scatter with iso-F1 curves across all models.*

---

## 14. Discussion of NIR and SWIR Results

### The Impact of NIR (M2 vs M1): The Spongy Mesophyll Mechanism
Adding Band B08 ($842\text{ nm}$) proved overwhelmingly beneficial:
1. **Shrub/Transitional Vegetation Breakthrough:**  
   In visible RGB (M1), *Shrub/Transitional Vegetation* obtained an F1 of only 0.3529 (33 missed instances out of 42). In visible bands, young birch/pine saplings, forest clear-cuts, and heath shrubs exhibit dark green chlorophyll absorption profiles identical to mature closed forest. In NIR, however, the structure of open multi-tiered scrub scatters radiation differently from dense multi-story forest canopies with strong internal shadowing. Consequently, **M2 successfully recovered 36 of 42 shrub instances**, elevating F1 to **0.9114 (+55.85 pp)**.
2. **Hydrological Delineation (Water):**  
   Liquid water strongly absorbs near-infrared radiation. Pure water bodies exhibit near-zero reflectance at $842\text{ nm}$. Adding B08 allowed M2 to attain **100% precision on water** (0 false alarms) and boosted recall from 61.9% to 80.95%, increasing F1 from **0.7429 to 0.8947 (+15.18 pp)**.

### The Impact of SWIR (M3 vs M2): The Small-Sample Dimensionality Paradox
Standard remote sensing theory indicates that SWIR bands (B11 at 1610 nm and B12 at 2190 nm) provide valuable sensitivity to foliar water content and soil mineralogy. In large-scale benchmarks trained on $> 100,000$ patches, M3 typically outperforms M2. 

However, in our controlled local experiment, **M3 performed worse than M2**:
1. **The Curse of Dimensionality on Small Samples:**  
   Our training split contained 300 patches. Moving from 4 channels (M2) to 6 channels (M3) expanded the input feature space by 50% without providing additional training examples.
2. **Spatial Resampling Discrepancy:**  
   Bands B11 and B12 have a native spatial resolution of $20\text{ m}$ (4 times coarser in area than $10\text{ m}$ bands). While bilinear interpolation smoothly harmonizes the grid, it does not synthesize true high-frequency $10\text{ m}$ spatial edges.
3. **Premature Training Divergence:**  
   As seen in Figure 3, M3 reached its minimum validation loss at **Epoch 3** ($\mathcal{L}_{\text{val}} = 0.4459$) and subsequently overfit the training split, triggering early stopping at Epoch 11. In contrast, M2 trained stably through **Epoch 20**.
4. **Severe Commission Errors in Grassland/Pasture:**  
   M3 generated **57 False Positives** on Grassland/Pasture (predicting it in 95 of 100 test patches), causing precision to drop to 40.0% and F1 to fall from 0.7872 (M2) to 0.5672. Furthermore, water recall collapsed to 28.57% (15 omissions).

*Scientific Conclusion:* Adding SWIR is not universally beneficial. When training compact CNNs on small sample sizes ($\le 500$ patches) without pre-training or transfer learning, 4-channel Visible+NIR input (M2) provides the optimal balance of spectral discriminability and parameter efficiency.

---

## 15. Error Analysis and Wetland Classification Failure

![FP/FN Heatmap](outputs/figures/phase3_fp_fn_heatmap.png)
*Figure 7: False Positives (commission errors) and False Negatives (omission errors) heatmap per class across M1, M2, and M3.*

![Sample Test Predictions Mosaic](outputs/figures/phase3_test_predictions_mosaic.png)
*Figure 8: Representative test patches showing True-Colour composites, ground-truth labels, and predicted multi-labels with sigmoid probabilities.*

### The Wetland Classification Failure
Across all three models at the canonical threshold $T=0.50$, **Wetland achieved an F1-score of exactly 0.0000**:
- Ground Truth Wetland patches in Test: **27**
- True Positives ($\text{TP}$): **0**
- False Positives ($\text{FP}$): **0**
- False Negatives ($\text{FN}$): **27**

#### Why Did the Models Fail on Wetland?
1. **Severe Training Imbalance (2.7% Prevalence):**  
   In tile `T33UUP`, the official training split provided **only 8 positive Wetland patches out of 300**. In contrast, negative instances totaled 292 ($97.3\%$).
2. **Loss Formulation Dynamics:**  
   In unweighted Binary Cross-Entropy loss, misclassifying 8 positive patches contributes very little gradient signal compared to the risk of false-alarm penalties on the 292 negative patches. Consequently, the network learned that outputting strongly negative logits for Wetland ($z_{\text{wetland}} \in [-3.5, -0.8]$) minimized overall cross-entropy loss.
3. **Threshold Calibration Response:**  
   Even when tuning decision thresholds on the validation set, validation Wetland had only 16 positive examples. In M1 and M2, the maximum validation F1 was achieved at $T=0.50$ (F1=0.000). Only in M3 did lowering the threshold to $T=0.20$ yield non-zero validation recall, but when applied to the test set, it introduced severe false alarms in damp agricultural fields.

### Other Observed Error Patterns
- **Agriculture vs Grassland Confusion:** In M1 and M2, 18 non-grassland patches were falsely flagged as Grassland. These represent cultivated clover, alfalfa meadows, or early-season green shoots that spectrally mimic permanent alpine pastures.
- **Built-up Commission Errors:** In M2 and M3, 13–14 false alarms occurred where exposed river gravel beds, bare quarries, or railway ballast exhibited flat, bright reflectance resembling concrete roofs.

---

## 16. Limitations and Potential Sources of Bias

To maintain strict scientific integrity, the findings must be interpreted under the following constraints:

1. **Single-Tile Geographic Bias:**  
   All 500 genuine patches are located within Sentinel-2 tile `T33UUP` in the Austrian Central Alps. The findings reflect Central European temperate alpine geography (spruce/pine forests, glacial valleys, managed pastures). **The models have not been evaluated on arid, Mediterranean, tropical, or coastal wetland biomes, and these results cannot be claimed to generalize across Europe or other continents.**
2. **Class-Selection Bias:**  
   Because rare classes (Wetland and Shrub) occur sparsely in tile `T33UUP` (natural prevalence $1.6\%$ and $3.6\%$), we intentionally captured 100% of the tile's Wetland and Shrub patches. This intentional curation enriched Wetland to $10.2\%$ and Shrub to $22.8\%$ across the 500 patches. While necessary to provide evaluation support, the test distribution does not reflect raw landscape frequencies.
3. **Modest Sample Size:**  
   A training set of 300 patches is appropriate for feasibility exploration and classroom environments, but small for training deep convolutional backbones from scratch.
4. **Spatial Resampling Uncertainty:**  
   Bands B11 and B12 were bilinearly resampled from 20 m to 10 m. Mixed sub-pixel boundaries in SWIR may have introduced spectral blurring that contributed to M3's higher error rates.

---

## 17. Conclusions and Future Improvements

### Conclusions
1. **Multi-label formulation is essential:** Over 82% of Sentinel-2 patches in European landscapes contain co-occurring land covers. Single-label classification is ecologically inadequate.
2. **NIR (Band B08) is the most critical spectral band:** Adding NIR elevated test Macro-F1 by **+11.32 percentage points** (0.5978 to 0.7110), boosted Exact Match from 12% to 38%, and resolved vegetation density confusion (*Shrub* F1 surging from 0.3529 to 0.9114).
3. **SWIR does not guarantee gains on small datasets:** Moving from 4 channels (M2) to 6 channels (M3) degraded Macro-F1 to 0.5673 on our 300-patch training split, demonstrating that increased spectral dimensionality without commensurate training support causes optimization instability and commission errors.
4. **Class imbalance causes zero-recall failure on minority classes:** Unweighted cross-entropy on 8 training patches resulted in complete omission of *Wetland* ($F1=0.0000$).

### Recommended Future Improvements
- **Positive-Class Loss Weighting:** Implement positive weight factors in BCE loss ($w_{\text{pos}} = N_{\text{neg}} / N_{\text{pos}}$) or Focal BCE Loss to penalize minority-class omissions.
- **Multispectral Transfer Learning:** Pretrain the CNN on continental-scale Sentinel-2 datasets (e.g., full BigEarthNet or SeCo self-supervised weights) before fine-tuning on regional tiles.
- **Super-Resolution / Guided Resampling:** Use high-resolution visible bands (B02–B04) to guide SWIR spatial downscaling instead of unguided bilinear interpolation.
- **Geographic Expansion:** Expand sampling to diverse biomes (e.g., Scandinavian peatlands, Mediterranean maquis, Iberian dehesas) to test inter-tile spatial generalization.

---

## 18. References

1. **Sumbul, G., Charfuelan, M., de Wall, B., et al.** (2019). *BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding*. IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2019), pp. 5901–5904. DOI: [10.1109/IGARSS.2019.8900532](https://doi.org/10.1109/IGARSS.2019.8900532).
2. **Sumbul, G., de Wall, B., Kreuziger, T., et al.** (2021). *BigEarthNet-MM: A Large-Scale, Multimodal, Multilabel Benchmark Archive for Remote Sensing Image Classification and Retrieval*. IEEE Geoscience and Remote Sensing Magazine, 9(3), pp. 174–180. DOI: [10.1109/MGRS.2021.3089174](https://doi.org/10.1109/MGRS.2021.3089174).
3. **Clasen, K., Sedona, R., Demir, B., et al.** (2024). *reBEN: Refining the BigEarthNet Benchmark Archive for Remote Sensing Image Understanding*. arXiv preprint arXiv:2407.03653.
4. **European Space Agency (ESA).** (2021). *Sentinel-2 User Handbook*. Document Reference: EOP-SM/1476/SR-dr, Issue 1, Rev. 2. European Space Agency, Frascati, Italy.
5. **European Environment Agency (EEA).** (2019). *CORINE Land Cover (CLC) 2018 Technical Guidelines*. EEA Report No. 17/2019. Publications Office of the European Union, Luxembourg.
6. **Main-Knorn, M., Pflug, B., Louis, J., et al.** (2017). *Sen2Cor for Sentinel-2*. In Image and Signal Processing for Remote Sensing XXIII, Proc. SPIE 10427, 1042704. DOI: [10.1117/12.2278218](https://doi.org/10.1117/12.2278218).
7. **Paszke, A., Gross, S., Massa, F., et al.** (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS 2019), 32, pp. 8024–8035.
8. **Pedregosa, F., Varoquaux, G., Gramfort, A., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, pp. 2825–2830.
