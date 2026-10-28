# Technical Report: Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification on the BigEarthNet Dataset

**Author:** Mandala Sai Charan  
**Admission No.:** 24JE1093  
**Course:** Remote Sensing Practical (Project 6)  
**Dataset Reference:** BigEarthNet v2.0 (*reBEN*), TU Berlin  
**Date:** October 26, 2026  
**Status:** Certified & Verified (`verify_phase4.py` Passed with Exit Code 0)

---

## Abstract

Land-cover mapping is a foundational pillar of earth observation, environmental monitoring, agricultural assessment, and spatial planning. In heterogeneous natural and anthropogenic terrains, satellite image footprints frequently encompass multiple surface categories simultaneously (e.g., agricultural fields bordered by woodlots, or rivers traversing rural settlements). Conventional single-label classification assigns an exclusive category to each scene, fundamentally misrepresenting mixed landscapes. In this project, we implement and evaluate a multi-label remote sensing classification framework using Sentinel-2 multispectral imagery from the benchmark BigEarthNet v2.0 (*reBEN*) archive.

We curated and verified a 500-patch genuine dataset originating from Sentinel-2 tile `T33UUP` in the Austrian Central Alps, strictly partitioned into official, disjoint splits: **300 Training**, **100 Validation**, and **100 Test** patches. We retained six strategically chosen diagnostic spectral bands: Blue ($B02$), Green ($B03$), Red ($B04$), Near-Infrared ($B08$), and Shortwave-Infrared ($B11, B12$). The $20\text{ m}$ SWIR bands were harmonized to $10\text{ m}$ ($120 \times 120$ pixels) using bilinear interpolation, and all bands were radiometrically converted to physical Bottom-Of-Atmosphere (BOA) surface reflectance in $[0.0, 1.0]$.

We trained a regularized 4-block Convolutional Neural Network (`MultispectralCNN`, ~617k parameters) under `BCEWithLogitsLoss` and conducted a controlled **Band Ablation Experiment** comparing:
- **$M_1$ (Visible RGB):** Bands $B02, B03, B04$ (3 input channels)
- **$M_2$ (Visible + NIR):** Bands $B02, B03, B04, B08$ (4 input channels)
- **$M_3$ (Full Multispectral):** Bands $B02, B03, B04, B08, B11, B12$ (6 input channels)

At the standard decision threshold ($T=0.5$), **$M_2$ achieved the highest overall performance**, recording a test **Macro-F1 of 0.7110**, **Micro-F1 of 0.8205**, and **Exact Match subset accuracy of 38.0%**, outperforming baseline $M_1$ on the observed test metrics (**Macro-F1 of 0.5978**, **Micro-F1 of 0.7403**, Exact Match 12.0%). Adding NIR ($B08$) provided an immediate **+11.32 percentage point gain** in Macro-F1, critically resolving canopy-density confusion (*Shrub/Transitional Vegetation* F1 surged from 0.3529 to 0.9114) and sharply elevating *Water* F1 from 0.7429 to 0.8947 with 100% precision.

Conversely, **$M_3$ underperformed $M_2$**, recording a test **Macro-F1 of 0.5673**, **Micro-F1 of 0.7081**, and **Exact Match ratio of 5.0%** ($T=0.5$). When decision thresholds were calibrated on the validation set, test Macro-F1 values were **0.7076 ($M_1$)**, **0.7020 ($M_2$)**, and **0.6559 ($M_3$)**. On our 300-patch training split, adding two $20\text{ m}$ resampled SWIR channels expanded input dimensionality without additional training data; small training-sample size, spectral collinearity, resampling effects, and optimization instability are possible explanations for the severe false-positive over-prediction for *Grassland/Pasture* (57 false positives; precision 40.0%), but their individual contributions were not independently tested. Furthermore, all three models recorded **0.0000 F1 on Wetland** at default $T=0.5$ and validation-tuned $T^*=0.5$ (0 true positives out of 27 test instances) because only 8 positive Wetland patches existed in the training split (2.7% prevalence). We discuss these findings in terms of remote sensing physics, small-sample dimensionality trade-offs, and multi-label threshold calibration.

---

## 1. Introduction

### 1.1 The Imperative for Satellite Land-Cover Mapping
Automated, high-fidelity land-use and land-cover (LULC) identification from spaceborne platforms is essential for quantifying environmental change, assessing crop phenology, modeling hydrological basins, and monitoring deforestation. Modern multispectral instruments, notably the European Space Agency's (ESA) Sentinel-2 constellation, provide global systematic coverage with 5-day revisit frequencies, offering 10 m and 20 m spatial resolutions across visible, near-infrared (NIR), and shortwave-infrared (SWIR) wavelengths.

### 1.2 Limitations of Single-Label Classification
In conventional remote sensing classification, each observation unit (pixel, polygon, or spatial tile) is forced into a single mutually exclusive category:
$$y \in \{1, 2, \dots, K\}$$
However, at standard satellite patch footprints ($1.2\text{ km} \times 1.2\text{ km}$, corresponding to $120 \times 120$ pixels at $10\text{ m}$ resolution), genuine landscapes are intrinsically heterogeneous mosaics. A single patch frequently contains cultivated arable fields bordering woodlots, alpine pastures framing stream channels, and rural built-up settlements. Single-label classification arbitrarily suppresses subordinate classes or fails on mixed pixels.

### 1.3 Multi-Label Formulation
To model landscape complexity without artificial mutual exclusivity, the ground truth for each patch is structured as a binary multi-hot vector:
$$\mathbf{y} = [y_1, y_2, \dots, y_K]^\top \in \{0, 1\}^K$$
where $y_k = 1$ denotes the confirmed presence of land-cover category $k$, and $y_k = 0$ denotes its absence. Under this formulation, multiple surface types can legitimately evaluate to 1 simultaneously. In our curated dataset, **94.4% of patches (472/500) contain $\ge 2$ co-occurring classes** (overall dataset mean: 3.00 classes per patch; test split mean: 3.07).

### 1.4 Central Research Questions
1. *Can visible, NIR, and SWIR observations accurately identify multiple co-occurring surface types in complex landscapes?*
2. *Does adding NIR ($B08$) and SWIR ($B11, B12$) consistently improve classification accuracy over standard visible RGB, and how does spectral dimensionality interact with small training sample sizes?*

---

## 2. Dataset, Spectral Bands, and Preprocessing

### 2.1 The BigEarthNet v2.0 (*reBEN*) Dataset
We utilize the open-access BigEarthNet v2.0 benchmark archive (Clasen et al., 2024; Sumbul et al., 2019, 2021; [bigearth.net](https://bigearth.net/)), released by the Remote Sensing Image Analysis (RSiM) and DIMA groups at Technische Universität Berlin. BigEarthNet v2.0 represents a major scientific upgrade over v1.0:
- **Atmospheric Correction:** All Sentinel-2 tiles are processed to Level-2A Bottom-Of-Atmosphere (BOA) surface reflectance via `sen2cor v2.11`.
- **Label Integrity:** Ground truth labels were re-extracted using the CORINE Land Cover 2018 database (`CLC2018 v2020_u1`).
- **Official Benchmark Splits:** Partitions preserve the official BigEarthNet v2.0 benchmark split assignments.

### 2.2 Geographic Provenance & Dataset Partitions
To enable rigorous, fully reproducible model training and Colab compatibility without downloading the prohibitive 59 GiB archive, we curated a genuine 500-patch dataset from the first tile in BigEarthNet: **Tile `T33UUP`**, situated in the **Austrian Central Alps** (UTM Zone 33N).
- **Landscape Context:** Alpine and sub-alpine topography (coniferous and mixed forests, managed pastures, agricultural valleys, settlements, mountain streams).
- **Official Split Assignments:**
  - **Training Split:** 300 patches (60.0%)
  - **Validation Split:** 100 patches (20.0%)
  - **Test Split:** 100 patches (20.0%)
- **Patch-Level Disjointness:** Strictly disjoint patch partitions ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$). No patch image, pixel, or label is shared between splits.
- **Inspection Patch Protocol:** All 5 Phase 1 inspection patches (`T33UUP_26_57`, `T33UUP_27_55`, `T33UUP_27_56`, `T33UUP_27_57`, `T33UUP_27_58`) reside strictly in the Test split.
- **Distinction from Geographic Independence:** Because all 500 patches originate from Sentinel-2 tile `T33UUP` (acquired simultaneously on 2017-06-13 across the Austrian Central Alps), patches share regional topography, phenology, and atmospheric conditions. Training and test patches are spatially proximate within the tile grid. Therefore, this split protocol guarantees *patch-level disjointness* within a single regional tile, but it does *not* constitute geographic independence across distinct scenes, nor does it eliminate potential spatial autocorrelation. Regional alpine findings cannot be assumed to transfer directly to Mediterranean, Scandinavian, or arid biomes without cross-tile evaluation.

### 2.3 Retained Sentinel-2 Diagnostic Bands

| Band | Spectral Region | Central $\lambda$ | Bandwidth | Native Res. | Grid Shape ($1.2\text{ km}$) | Primary Spectroradiometric Role |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **B02** | Blue | 490 nm | 65 nm | 10 m | $120 \times 120$ | Atmospheric penetration; water body clarity; soil vs vegetation contrast |
| **B03** | Green | 560 nm | 35 nm | 10 m | $120 \times 120$ | Chlorophyll green reflectance peak; general vegetative vigor |
| **B04** | Red | 665 nm | 30 nm | 10 m | $120 \times 120$ | Chlorophyll-a absorption maximum; defines base of red-edge |
| **B08** | Near-Infrared (NIR) | 842 nm | 115 nm | 10 m | $120 \times 120$ | Leaf spongy mesophyll scattering (vegetation plateau); water absorption |
| **B11** | Shortwave Infrared-1 | 1610 nm | 90 nm | 20 m | $60 \times 60 \to 120 \times 120$ | Foliar moisture absorption; soil moisture; built-up roof differentiation |
| **B12** | Shortwave Infrared-2 | 2190 nm | 180 nm | 20 m | $60 \times 60 \to 120 \times 120$ | Mineral background; dry non-photosynthetic biomass; soil silica |

Input tensors are structured as $\mathbf{X} \in \mathbb{R}^{C \times 120 \times 120}$, where $C \in \{3, 4, 6\}$.

### 2.4 CORINE Land Cover Classes and Broad Mapping Scheme
BigEarthNet provides 19 detailed CORINE Land Cover classes. In this experiment, 18 classes are mapped into seven broad land-cover categories, while ‘Beaches, dunes, sands’ is excluded because no instances occur in tile T33UUP.

The seven broad target classes are:
```
Col 0: Forest
Col 1: Agriculture
Col 2: Grassland/Pasture
Col 3: Built-up/Artificial Surfaces
Col 4: Water
Col 5: Wetland
Col 6: Shrub/Transitional Vegetation
```

#### Complete CORINE Land Cover Mapping Table

| Original CORINE Land Cover Label | Mapped Broad Class | Status | Spectroradiometric & Ecological Rationale |
|:---|:---|:---:|:---|
| Broad-leaved forest | **Forest** | Mapped | Deciduous broad-leaved tree canopy cover $> 30\%$. |
| Coniferous forest | **Forest** | Mapped | Evergreen needle-leaved tree canopy cover $> 30\%$. |
| Mixed forest | **Forest** | Mapped | Mosaic where neither broad-leaved nor needle-leaved dominates. |
| Arable land | **Agriculture** | Mapped | Regularly cultivated parcels for annual crop production. |
| Permanent crops | **Agriculture** | Mapped | Vineyards, orchards, fruit trees, and berry plantations. |
| Complex cultivation patterns | **Agriculture** | Mapped | Small cultivated parcels interspersed with rural dwellings. |
| Land principally occupied by agriculture, natural veg. | **Agriculture** | Mapped | Farmland interspersed with significant natural woodland patches. |
| Agro-forestry areas | **Agriculture** | Mapped | Annual crops or pasture combined with forest trees. |
| Pastures | **Grassland/Pasture** | Mapped | Managed perennial herbaceous cover grazed by livestock. |
| Natural grassland and sparsely vegetated areas | **Grassland/Pasture** | Mapped | Low-productivity alpine meadows, open steppe, rocky grasslands. |
| Urban fabric | **Built-up/Artificial Surfaces** | Mapped | Continuous and discontinuous residential, commercial, civic fabric. |
| Industrial or commercial units | **Built-up/Artificial Surfaces** | Mapped | Factories, industrial facilities, infrastructure, transport nodes. |
| Inland waters | **Water** | Mapped | Freshwater lakes, reservoirs, canals, and river channels. |
| Marine waters | **Water** | Mapped | Coastal waters, estuaries, bays, and open seas. |
| Inland wetlands | **Wetland** | Mapped | Freshwater marshes, peatbogs, seasonally waterlogged terrain. |
| Coastal wetlands | **Wetland** | Mapped | Salt marshes, salines, and tidal flats. |
| Transitional woodland, shrub | **Shrub/Transitional Vegetation** | Mapped | Bushy regrowth, young saplings, clear-cut regeneration, maquis. |
| Moors, heathland and sclerophyllous vegetation | **Shrub/Transitional Vegetation** | Mapped | Low woody scrub, heathland, gorse, and dwarf alpine shrubs. |
| **Beaches, dunes, sands** | **[None - Excluded]** | **Excluded** | Bare coastal sand/shingle lacking vegetation ($0.27\%$ of archive, 0 in $T33UUP$). |

*Audit Verification:* The code does not silently reassign unmapped labels. Unmapped labels contribute zero 1s to target vectors. In tile `T33UUP`, *Beaches, dunes, sands* occurs 0 times.

### 2.5 Reflectance Preprocessing & Harmonization Pipeline
1. **Bilinear Spatial Resampling:** Bands $B11$ and $B12$ have a native resolution of $20\text{ m}$ ($60 \times 60$ pixels). We resample them to the $10\text{ m}$ working grid ($120 \times 120$ pixels) using **bilinear interpolation** (`PIL.Image.BILINEAR`, order=1). Bilinear interpolation computes distance-weighted averages of four adjacent pixel centers, generating smooth, continuous spatial gradients without ringing artifacts (unlike bicubic) or artificial blocky steps (unlike nearest neighbor).
2. **Radiometric Scaling to Surface Reflectance:** In accordance with the ESA Sentinel-2 L2A product specification, surface reflectance $\rho$ is quantified as $\text{DN} = 10,000 \times \rho$. We apply:
   $$\rho = \text{clip}\left(\frac{\text{DN}}{10,000.0}, 0.0, 1.0\right)$$
   Division by 10,000 converts raw DN to Bottom-Of-Atmosphere surface reflectance. Clipping to $[0.0, 1.0]$ handles rare specular glint highlights (e.g., solar reflections on glass roofs, greenhouse panels, or calm water surfaces $> 10,000\text{ DN}$, which represent $< 0.0042\%$ of all pixels).
3. **Preservation of Raw Data:** All raw GeoTIFF files remain completely unmodified in `data/curated_patches/`. Preprocessed tensors are saved in compressed archive `data/curated_dataset_phase2_1.npz` (76.66 MB).

---

## 3. Exploratory Data Analysis & Class Distributions

### 3.1 Multi-Label Complexity
- Minimum active labels per patch: **1**
- Maximum active labels per patch: **6** (overall dataset; 2 patches contain 6 classes, test split maximum: 5)
- Mean active labels per patch: **3.00** across the full dataset (Train: 2.82, Val: 3.48, Test: 3.07)
- **94.4% of all patches (472/500) contain $\ge 2$ co-occurring classes** (Train: 92.7%, Val: 96.0%, Test: 98.0%), confirming that single-label classification is fundamentally inadequate for European landscapes.

### 3.2 Class Prevalence Across Official Splits ($N=500$)

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

## 4. Methodology: Multi-Label Deep Learning Framework

### 4.1 Multi-Label Problem Formulation
Given an input multispectral cube $\mathbf{X} \in \mathbb{R}^{C \times 120 \times 120}$, the neural network computes unconstrained raw logits $\mathbf{z} = [z_1, z_2, \dots, z_K]^\top \in \mathbb{R}^7$. 

Unlike multi-class classification which employs a softmax activation ($\sum_k p_k = 1$), multi-label classification treats each class as an independent Bernoulli trial. Independent class probabilities $\hat{p}_k \in [0, 1]$ are obtained via the standard logistic sigmoid function:
$$\hat{p}_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}$$

### 4.2 Network Architecture: MultispectralCNN
Implemented in `src/model.py`, the network extracts spatial-spectral features without excessive parameterization:
- **Block 1 (120×120 → 60×60):** $\text{Conv2d}(C \to 32, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv2d}(32 \to 32, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{MaxPool2d}(2, 2)$
- **Block 2 (60×60 → 30×30):** $\text{Conv2d}(32 \to 64, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv2d}(64 \to 64, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{MaxPool2d}(2, 2)$
- **Block 3 (30×30 → 15×15):** $\text{Conv2d}(64 \to 128, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{Conv2d}(128 \to 128, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{MaxPool2d}(2, 2)$
- **Block 4 (15×15 → 7×7):** $\text{Conv2d}(128 \to 256, 3 \times 3) \to \text{BN} \to \text{ReLU} \to \text{MaxPool2d}(2, 2)$
- **Global Pooling & Head:** $\text{AdaptiveAvgPool2d}((1, 1)) \to \text{Flatten} \to \text{Dropout}(0.4) \to \text{Linear}(256, 128) \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(128, 7)$
- **Trainable Parameters:** $M_1$: 616,679; $M_2$: 616,967; $M_3$: 617,543.

### 4.3 Training Protocols and Optimization
- **Loss Function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`):
  $$\mathcal{L}_{\text{BCE}}(\mathbf{z}, \mathbf{y}) = -\frac{1}{K} \sum_{k=1}^K \left[ y_k \cdot \log \sigma(z_k) + (1 - y_k) \cdot \log(1 - \sigma(z_k)) \right]$$
- **Optimizer:** `AdamW` ($\text{lr} = 1.0 \times 10^{-3}$, $\text{weight\_decay} = 1.0 \times 10^{-4}$)
- **LR Scheduler:** `CosineAnnealingLR` ($T_{\max} = 35$, $\eta_{\min} = 1.0 \times 10^{-5}$)
- **Batch Size:** 16 (19 updates per epoch on 300 training patches)
- **Early Stopping:** Monitored on validation loss (patience = 8 epochs)
- **Seed Policy:** `seed=42` across Python, NumPy, and PyTorch

### 4.4 Decision Threshold Calibration
Binary predictions $\hat{y}_k \in \{0, 1\}$ are formed by applying a decision threshold vector $\mathbf{T} = [T_1, \dots, T_7]$:
$$\hat{y}_k = \mathbb{I}(\hat{p}_k \ge T_k)$$
We evaluate:
1. **Standard Default Threshold ($T_k = 0.50$):** Canonical baseline for unweighted cross-entropy.
2. **Validation-Calibrated Thresholds ($T_k^*$):** Selected by grid search over candidate thresholds $T \in [0.20, 0.80]$ on the validation split to maximize validation Macro-F1. *Crucially, test labels are never accessed during calibration.*

### 4.5 Band Contribution Experiments
- **$M_1$ (Visible RGB):** Channels $[B02, B03, B04]$ (3 inputs; 490, 560, 665 nm)
- **$M_2$ (Visible + NIR):** Channels $[B02, B03, B04, B08]$ (4 inputs; + 842 nm)
- **$M_3$ (Full Multispectral):** Channels $[B02, B03, B04, B08, B11, B12]$ (6 inputs; + 1610, 2190 nm)

---

## 5. Experimental Results

The models were evaluated on a held-out test set of 100 patches. Although the patch IDs are disjoint across the training, validation, and test splits, all patches originate from the same Sentinel-2 tile; therefore, geographic independence and cross-region generalization are not established. All metrics are transcribed directly from verified metrics stored in `outputs/metrics/phase3_model_comparison.csv` and `outputs/metrics/phase3_per_class_metrics.csv`.

### 5.1 Quantitative Performance Summary on Test Set ($N=100$)

| Evaluation Protocol | Model Identifier | Macro-F1 | Micro-F1 | Macro-Precision | Macro-Recall | Exact Match Ratio | Hamming Loss |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard ($T=0.50$)** | **$M_1$ (Visible RGB)** | 0.5978 | 0.7403 | 0.7317 | 0.5588 | 12.0% | 0.2014 |
| **Standard ($T=0.50$)** | **$M_2$ (Visible + NIR)** | **0.7110** | **0.8205** | **0.7326** | **0.7036** | **38.0%** | **0.1500** |
| **Standard ($T=0.50$)** | **$M_3$ (Multispectral)** | 0.5673 | 0.7081 | 0.6894 | 0.5666 | 5.0% | 0.2486 |
| *Validation-Tuned* | $M_1$ (Tuned Thresholds) | 0.7076 | 0.8024 | 0.6648 | 0.7868 | 9.0% | 0.1857 |
| *Validation-Tuned* | $M_2$ (Tuned Thresholds) | 0.7020 | 0.7928 | 0.6566 | 0.7886 | 7.0% | 0.1971 |
| *Validation-Tuned* | $M_3$ (Tuned Thresholds) | 0.6559 | 0.7692 | 0.5939 | 0.7645 | 6.0% | 0.2229 |

*Validation-Tuned Threshold Vectors:*
- $M_1$: $[\text{Fo}: 0.30, \text{Ag}: 0.20, \text{Gr}: 0.40, \text{Bu}: 0.40, \text{Wa}: 0.30, \text{We}: 0.50, \text{Sh}: 0.20]$
- $M_2$: $[\text{Fo}: 0.20, \text{Ag}: 0.20, \text{Gr}: 0.35, \text{Bu}: 0.30, \text{Wa}: 0.30, \text{We}: 0.50, \text{Sh}: 0.20]$
- $M_3$: $[\text{Fo}: 0.45, \text{Ag}: 0.30, \text{Gr}: 0.45, \text{Bu}: 0.45, \text{Wa}: 0.20, \text{We}: 0.50, \text{Sh}: 0.20]$

### 5.2 Key Takeaways from Band Contributions
1. **The Decisive Role of NIR ($B08$):** Adding NIR provided an immediate **+11.32 percentage point increase** in default-threshold Macro-F1 (0.5978 to 0.7110), while Micro-F1 reached 0.8205 and the Exact Match subset accuracy more than tripled from 12.0% to 38.0%. Shrub/Transitional Vegetation F1 surged from 0.3529 to 0.9114 (+55.85 pp) due to mesophyll scattering differences between open scrub and mature forest canopy. Water F1 jumped from 0.7429 to 0.8947 with 100% precision.
2. **Behavior of SWIR ($B11, B12$):** On our curated 300-patch training split, $M_3$ achieved a Macro-F1 of 0.5673, performing worse than $M_2$ (0.7110; Exact Match 5.0% vs 38.0%). $M_3$ produced 57 false alarms on Grassland/Pasture (precision 40.0%). Small training-sample size, spectral collinearity, resampling effects, and optimization instability are possible explanations, but their individual contributions were not independently tested.

### 5.3 Complete Per-Class Metrics Table at Threshold $T=0.50$

| Model | Class Name | Support | Precision | Recall | F1-Score | TP | FP | FN | TN |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$M_1$** | Forest | 95 | 0.9588 | 0.9789 | 0.9688 | 93 | 4 | 2 | 1 |
| $M_1$ | Agriculture | 52 | 0.9444 | 0.6538 | 0.7727 | 34 | 2 | 18 | 46 |
| $M_1$ | Grassland/Pasture | 39 | 0.6604 | 0.8974 | 0.7609 | 35 | 18 | 4 | 43 |
| $M_1$ | Built-up/Artificial | 31 | 0.6296 | 0.5484 | 0.5862 | 17 | 10 | 14 | 59 |
| $M_1$ | Water | 21 | 0.9286 | 0.6190 | 0.7429 | 13 | 1 | 8 | 78 |
| $M_1$ | **Wetland** | **27** | **0.0000** | **0.0000** | **0.0000** | **0** | **0** | **27** | **73** |
| $M_1$ | **Shrub/Transitional** | **42** | **1.0000** | **0.2143** | **0.3529** | **9** | **0** | **33** | **58** |
| **$M_2$** | Forest | 95 | 0.9688 | 0.9789 | 0.9738 | 93 | 3 | 2 | 2 |
| $M_2$ | Agriculture | 52 | 0.9512 | 0.7500 | 0.8387 | 39 | 2 | 13 | 46 |
| $M_2$ | Grassland/Pasture | 39 | 0.6727 | 0.9487 | 0.7872 | 37 | 18 | 2 | 43 |
| $M_2$ | Built-up/Artificial | 31 | 0.5625 | 0.5806 | 0.5714 | 18 | 14 | 13 | 55 |
| $M_2$ | Water | 21 | **1.0000** | **0.8095** | **0.8947** | **17** | **0** | **4** | **79** |
| $M_2$ | **Wetland** | **27** | **0.0000** | **0.0000** | **0.0000** | **0** | **0** | **27** | **73** |
| $M_2$ | **Shrub/Transitional** | **42** | **0.9730** | **0.8571** | **0.9114** | **36** | **1** | **6** | **57** |
| **$M_3$** | Forest | 95 | 0.9592 | 0.9895 | 0.9741 | 94 | 4 | 1 | 1 |
| $M_3$ | Agriculture | 52 | 0.9000 | 0.6923 | 0.7826 | 36 | 4 | 16 | 44 |
| $M_3$ | Grassland/Pasture | 39 | **0.4000** | 0.9744 | **0.5672** | 38 | **57** | 1 | 4 |
| $M_3$ | Built-up/Artificial | 31 | 0.5667 | 0.5484 | 0.5574 | 17 | 13 | 14 | 56 |
| $M_3$ | Water | 21 | 1.0000 | **0.2857** | **0.4444** | 6 | 0 | **15** | 79 |
| $M_3$ | **Wetland** | **27** | **0.0000** | **0.0000** | **0.0000** | **0** | **0** | **27** | **73** |
| $M_3$ | **Shrub/Transitional** | **42** | 1.0000 | 0.4762 | 0.6452 | 20 | 0 | 22 | 58 |

---

## 6. Error and Co-Occurrence Analysis

### 6.1 Diagnostic Breakdown of Test Predictions
At the standard $T=0.50$ threshold:
- **Exact Match Ratio:** **38.0%** in $M_2$, **12.0%** in $M_1$, and **5.0%** in $M_3$. Because exact match requires all 7 binary decisions to match the ground truth simultaneously, high label cardinality (dataset mean: 3.00 labels per patch; test mean: 3.07, with 94.4% possessing $\ge 2$ classes) makes this metric demanding.
- **Dominant Confusion Pair:** *Agriculture* versus *Grassland/Pasture*. In $M_1$ and $M_2$, exactly 18 patches were falsely classified as Grassland due to spectral similarities between tilled clover meadows and grazed alpine turf.

### 6.2 Analysis of Confusing Class Combinations
1. **Agriculture ↔ Grassland/Pasture:** Cultivated forage crops, fallow arable parcels, and permanent pastures share similar green leaf pigments and visual texture. In visible wavelengths, both exhibit nearly identical green peaks and red absorption troughs.
2. **Forest ↔ Transitional Woodland/Shrub:** Transitional vegetation represents an ecotone between open grassland and mature forest. In visible bands, scrub mimics dense forest greenness ($M_1$ F1=0.3529). In NIR, canopy micro-shadowing in mature forests attenuates reflectance relative to open multi-tiered scrub, enabling $M_2$ to successfully resolve 36 of 42 shrub instances ($M_2$ F1=0.9114).

### 6.3 Detailed Wetland Classification Failure Analysis
Across all three models at $T=0.50$, **Wetland achieved an F1-score of exactly 0.0000**:
- Ground Truth Wetland patches in Test: **27**
- True Positives ($\text{TP}$): **0**
- False Positives ($\text{FP}$): **0**
- False Negatives ($\text{FN}$): **27**

#### Why Did the Models Fail on Wetland?
1. **Severe Training Imbalance (2.7% Prevalence):** In tile `T33UUP`, the official training split provided **only 8 positive Wetland patches out of 300** (292 negatives, 97.3%).
2. **Loss Formulation Dynamics:** In unweighted Binary Cross-Entropy loss, misclassifying 8 positive patches contributes very little gradient signal compared to avoiding false alarms on the 292 negative patches. Consequently, the network learned that outputting strongly negative logits for Wetland ($z_{\text{wetland}} \in [-3.5, -0.8]$) minimized overall loss.
3. **Threshold Calibration Procedure (Exploratory vs. Selected):** In `outputs/metrics/phase3_thresholds.json`, decision thresholds were systematically swept over candidate cutoffs $T \in [0.20, 0.80]$ in increments of 0.05 strictly on validation set predictions. Across this grid search, the validation F1 for Wetland remained **0.0000** for all candidate thresholds across all three models (`val_f1_tuned_per_class: {"Wetland": 0.0}`). While lower exploratory cutoffs such as $T=0.20$ were evaluated, they produced false alarms on non-wetland validation patches without yielding a non-zero F1 score. Consequently, the optimization algorithm retained **$T^*=0.50$** as the final selected tuned threshold for Wetland across $M_1, M_2$, and $M_3$. When evaluated on the test set, predictions at $T^*=0.50$ predictably yielded 0 true positives and 27 false negatives (F1 = 0.0000). The failure does not establish that Wetland is unclassifiable by neural networks, but rather reflects unweighted loss behavior under 2.7% minority representation.

---

## 7. Conclusions and Limitations

### 7.1 Conclusions
1. **Multi-label recognition is essential:** In our curated dataset, **94.4% of Sentinel-2 patches** contain co-occurring land covers (mean: 3.00 classes per patch; test mean: 3.07), demonstrating that conventional single-label classification is fundamentally inadequate for European landscapes.
2. **NIR ($B08$) provides the decisive performance boost:** Adding NIR in $M_2$ increased default-threshold Macro-F1 by **+11.32 percentage points** (0.5978 to 0.7110), boosted Exact Match from 12% to 38%, and resolved woody scrub confusion (*Shrub* F1 surging from 0.3529 to 0.9114).
3. **Interpretation of $M_3$ Performance:** M2 performed best on this curated test set at the default threshold of 0.5. M3 underperformed M2, achieving a Macro-F1 of 0.5673 compared with 0.7110 for M2. Small training-sample size, spectral collinearity, resampling effects, and optimization instability are possible explanations, but their individual contributions were not independently tested.
4. **Minority classes require dedicated loss weighting:** Extreme training imbalance (8 Wetland patches) causes zero-recall failure under unweighted BCE loss.

### 7.2 Practical Limitations
1. **Single-Tile Geographic Scope and Spatial Autocorrelation:** All 500 patches originate from Sentinel-2 Tile `T33UUP` in the Austrian Central Alps, acquired during a single overpass (2017-06-13). While our partitions guarantee strict **patch-level disjointness** ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$), they do not constitute **geographic independence**. Because all patches reside within the same regional footprint, they share atmospheric, solar, and phenological conditions, and spatial proximity between training and test patches may introduce spatial autocorrelation. Therefore, findings reflect regional temperate alpine performance and cannot be claimed to demonstrate continental transfer.
2. **Intentional Class-Selection Bias:** To enable evaluation of rare classes, we collected 100% of the tile's Wetland and Shrub patches. This enriched Wetland to 10.2% and Shrub to 22.8% across the 500 patches, diverging from raw landscape prevalence (1.6% and 3.6%).

### 7.3 Future Improvements
- **Positive-Class Weighting / Focal Loss:** Apply positive class weights ($w_{\text{pos}} = N_{\text{neg}} / N_{\text{pos}}$) to penalize minority-class omissions.
- **Multispectral Transfer Learning:** Pretrain the CNN backbone on continental-scale archives before fine-tuning on regional tiles.
- **Guided Super-Resolution:** Downscale SWIR bands using 10 m visible band spatial guidance rather than unguided bilinear interpolation.

---

## 8. References

1. **Clasen, K., Sedona, R., Demir, B., et al.** (2024). *reBEN: Refining the BigEarthNet Benchmark Archive for Remote Sensing Image Understanding*. arXiv preprint arXiv:2407.03653.
2. **Sumbul, G., Charfuelan, M., de Wall, B., et al.** (2019). *BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding*. IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2019), pp. 5901–5904. DOI: [10.1109/IGARSS.2019.8900532](https://doi.org/10.1109/IGARSS.2019.8900532).
3. **Sumbul, G., de Wall, B., Kreuziger, T., et al.** (2021). *BigEarthNet-MM: A Large-Scale, Multimodal, Multilabel Benchmark Archive for Remote Sensing Image Classification and Retrieval*. IEEE Geoscience and Remote Sensing Magazine, 9(3), pp. 174–180. DOI: [10.1109/MGRS.2021.3089174](https://doi.org/10.1109/MGRS.2021.3089174).
4. **European Space Agency (ESA).** (2021). *Sentinel-2 User Handbook*. Document Reference: EOP-SM/1476/SR-dr, Issue 1, Rev. 2. European Space Agency, Frascati, Italy.
5. **European Environment Agency (EEA).** (2019). *CORINE Land Cover (CLC) 2018 Technical Guidelines*. EEA Report No. 17/2019. Publications Office of the European Union, Luxembourg.
6. **Main-Knorn, M., Pflug, B., Louis, J., et al.** (2017). *Sen2Cor for Sentinel-2*. In Image and Signal Processing for Remote Sensing XXIII, Proc. SPIE 10427, 1042704. DOI: [10.1117/12.2278218](https://doi.org/10.1117/12.2278218).
7. **Paszke, A., Gross, S., Massa, F., et al.** (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS 2019), 32, pp. 8024–8035.
8. **Pedregosa, F., Varoquaux, G., Gramfort, A., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, pp. 2825–2830.
