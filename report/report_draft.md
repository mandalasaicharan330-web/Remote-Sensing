# Technical Report: Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification on the BigEarthNet Dataset

**Name:** Mandala Sai Charan  
**Admission No.:** 24JE1093  
**Course:** Remote Sensing Practical (Project 6)  
**Dataset Reference:** BigEarthNet v2.0 (*reBEN*), TU Berlin  

---

## Abstract
Land-cover mapping is a cornerstone of earth observation, environmental monitoring, agricultural yield assessment, and natural hazard management. In heterogeneous natural and anthropogenic terrains, satellite image patches commonly encompass multiple surface categories simultaneously (e.g., agricultural fields bordered by woodlots, or rivers traversing built-up settlements). Conventional single-label classification assigns an exclusive category to each scene, fundamentally misrepresenting mixed landscapes. In this project, we implement and evaluate a multi-label remote sensing classification framework using Sentinel-2 multispectral imagery from the benchmark BigEarthNet v2.0 (*reBEN*) dataset. By retaining six strategically chosen spectral bands—Blue ($B2$), Green ($B3$), Red ($B4$), Near-Infrared ($B8$), and Short-Wave Infrared ($B11, B12$)—we investigate whether optical and infrared observations resolve co-occurring land-cover categories. We design controlled band-ablation experiments comparing visible RGB ($M_1$), visible plus NIR ($M_2$), and full multispectral inputs ($M_3$). Our findings show that Near-Infrared provides the greatest boost in recognizing vegetative canopy and water bodies, while SWIR aids in differentiating soil, moisture, and transitional shrubland. We also analyze the persistent confusions between spectrally similar co-occurring classes, notably Agriculture versus Grassland/Pasture and Forest versus Transitional Vegetation.

---

## 1. Introduction

### 1.1 The Imperative for Satellite Land-Cover Mapping
Accurate, automated land-use and land-cover (LULC) identification from spaceborne sensors is essential for tracking urbanization, deforestation, crop phenology, and water resource distribution. Multispectral sensors aboard platforms such as the European Space Agency's (ESA) Sentinel-2 constellation offer high radiometric resolution, 5-day revisit cycles, and broad coverage across visible, near-infrared (NIR), and short-wave infrared (SWIR) wavelengths.

### 1.2 Limitations of Single-Label Classification
In standard remote sensing classification, each spatial unit (tile, parcel, or patch) is forced into a single mutually exclusive category:
$$y \in \{1, 2, \dots, K\}$$
However, at typical patch dimensions ($1.2\text{ km} \times 1.2\text{ km}$, corresponding to $120 \times 120$ pixels at $10\text{ m}$ spatial resolution), genuine earth surfaces are inherently heterogeneous. Within a single patch, one frequently observes cultivated fields adjacent to farmhouse clusters, a stream flowing past pastures, or dense woodland fringing scrubland. Single-label approaches arbitrarily suppress subordinate land types or fail entirely on mixed pixels.

### 1.3 Multi-Label Formulation
To model real-world complexity, the target for each satellite patch is formulated as a binary multi-hot vector:
$$\mathbf{y} = [y_1, y_2, \dots, y_K], \quad y_k \in \{0, 1\}$$
where $y_k = 1$ denotes the presence of land-cover category $k$, and $y_k = 0$ denotes its absence. Under this formulation, multiple labels can simultaneously evaluate to 1.

### 1.4 Central Research Questions
This investigation addresses two core questions:
1. *Can visible, NIR, and SWIR Sentinel-2 observations accurately identify multiple co-occurring land-cover categories within the same image patch?*
2. *Which classes are most prone to mutual confusion when they appear together, and does infrared information alleviate this confusion?*

---

## 2. Dataset and Reference Labels

### 2.1 The BigEarthNet v2.0 (reBEN) Dataset
We employ the open BigEarthNet v2.0 benchmark archive (Clasen et al., 2024; [bigearth.net](https://bigearth.net/)), released by the Remote Sensing Image Analysis (RSiM) and DIMA groups at Technische Universität Berlin. 

BigEarthNet v2.0 represents a major upgrade over the legacy v1.0 release:
- **Atmospheric Correction:** All Sentinel-2 tiles were atmospherically corrected to surface reflectance (Level-2A Bottom-Of-Atmosphere, BOA) using `sen2cor v2.11`.
- **Label Integrity:** Ground truth labels were re-extracted using the most recent CORINE Land Cover 2018 database (`CLC2018 v2020_u1`), overcoming label noise and misregistration found in v1.0.
- **Geographical Splits:** Splits are assigned geographically to ensure spatial independence between training, validation, and testing tiles, preventing data leakage.
- **Archive Size:** The complete Sentinel-2 archive comprises 549,488 patches spanning 10 European nations, totaling ~58.9 GiB compressed (`BigEarthNet-S2.tar.zst`).

### 2.2 Selected Sentinel-2 Spectral Bands
Consistent with project requirements, six diagnostic Sentinel-2 bands were retained:

| Band | Spectral Region | Central Wavelength ($\lambda$) | Native Resolution | Physical Remote Sensing Relevance |
| :---: | :---: | :---: | :---: | :--- |
| **B2** | Blue | $490\text{ nm}$ | $10\text{ m}$ | Atmospheric penetration; water body discrimination; soil reflection |
| **B3** | Green | $560\text{ nm}$ | $10\text{ m}$ | Peak reflectance of green vegetation chlorophyll; turbidity |
| **B4** | Red | $665\text{ nm}$ | $10\text{ m}$ | Maximum chlorophyll absorption; contrast against bare soil |
| **B8** | NIR (Broad) | $842\text{ nm}$ | $10\text{ m}$ | Cell structure scattering (NIR plateau); canopy biomass; water absorption |
| **B11** | SWIR-1 | $1610\text{ nm}$ | $20\text{ m}$ | Canopy water content; soil moisture; built-up/mineral reflectance |
| **B12** | SWIR-2 | $2190\text{ nm}$ | $20\text{ m}$ | Dry biomass; senescent vegetation; geological and built structures |

The multispectral input tensor is structured as:
$$\mathbf{X} = [B2, B3, B4, B8, B11, B12] \in \mathbb{R}^{6 \times 120 \times 120}$$

### 2.3 Simplified Broad Class Scheme
BigEarthNet provides 19 detailed CORINE classes. To maintain computational tractability for an undergraduate practical while preserving spectral interpretability, we mapped the 19 classes into **7 broad land-cover categories**:

```
1. Forest:
   └── Broad-leaved forest, Coniferous forest, Mixed forest
2. Agriculture:
   └── Arable land, Permanent crops, Complex cultivation patterns,
       Land principally occupied by agriculture with natural vegetation, Agro-forestry areas
3. Grassland/Pasture:
   └── Pastures, Natural grassland and sparsely vegetated areas
4. Built-up / Artificial Surfaces:
   └── Urban fabric, Industrial or commercial units
5. Water:
   └── Inland waters, Marine waters
6. Wetland:
   └── Inland wetlands, Coastal wetlands
7. Shrub / Transitional Vegetation:
   └── Transitional woodland/shrub, Moors, heathland and sclerophyllous vegetation
```
*Note on Minor Classes:* "Beaches, dunes, sands" occurs in fewer than 0.28% of patches and does not cleanly map into these 7 categories; it is preserved in the metadata but excluded from the active target vector to maintain ground-truth consistency.

---

## 3. Spectral and Label Analysis

### 3.1 Multispectral False-Colour Composite Analysis
To interpret the multispectral response visually before feeding raw arrays to a neural network, three composite types were constructed:
1. **True-Colour Composite ($B4-B3-B2$):** Renders natural visual appearances. Agricultural fields show pale greens and yellows; forest canopies appear deep green; built-up roofs appear gray/white; deep water appears dark blue.
2. **Color-Infrared (CIR) Composite ($B8-B4-B3$):** NIR is assigned to the red channel, Red to green, and Green to blue. Because green foliage reflects strongly in NIR, dense forests and vigorous crops appear intensely bright red. Deep water absorbs NIR radiation completely and appears pitch black, offering unambiguous coastline and bank delineation.
3. **SWIR False-Colour Composite ($B12-B8-B4$):** Assigning SWIR-2 ($B12$) to red highlights soil moisture gradients and differentiates bare dry ground from moist vegetative cover.

### 3.2 Label Co-Occurrence Dynamics
Inspection of the official `metadata.parquet` (480,038 valid patches) reveals that **82.01% of all patches contain 2 or more distinct land-cover labels**:
- Single-label patches: 17.99%
- 2 co-occurring labels: 19.72%
- 3 co-occurring labels: 28.31%
- 4 co-occurring labels: 21.26%
- $\ge 5$ co-occurring labels: 12.72%

The label co-occurrence matrix $C_{ij} = \sum_n y_{n,i} y_{n,j}$ highlights frequent ecological pairings:
- **Agriculture + Grassland/Pasture:** Highly frequent due to mixed pastoral-arable agricultural farming across Europe.
- **Forest + Shrub/Transitional Vegetation:** Characteristic of woodland margins, selective harvesting cuts, and hillside scrub transitions.
- **Agriculture + Built-up:** Reflecting peri-urban zones and rural agricultural settlements.

### 3.3 Mean Spectral Signatures
For each broad class $C_k$, the mean spectral response is defined as:
$$\bar{\mathbf{s}}_k = [\bar{B2}, \bar{B3}, \bar{B4}, \bar{B8}, \bar{B11}, \bar{B12}]$$
Spectral signatures display pronounced domain-specific features:
- **Forest:** Exhibits classic chlorophyll absorption in $B4$ ($\sim 400$), a sharp "red-edge" transition to $B8$ ($\sim 3,600$), and moderate SWIR reflectance ($B11 \sim 1,900, B12 \sim 1,000$).
- **Water:** High visible backscatter ($B2, B3$), followed by steep absorption towards zero in NIR ($B8 < 300$) and SWIR ($B11, B12 < 150$).
- **Built-up:** Relatively flat, monotonic upward slope across visible into SWIR, with no characteristic red-edge peak.
- **Agriculture vs. Grassland:** Both display a vegetative red-edge, but pasture generally exhibits higher moisture and slightly lower red absorption depending on seasonal grazing.

---

## 4. Methodology

### 4.1 Data Access & On-the-Fly Streaming
To comply with computational restrictions and avoid downloading the massive 59 GiB archive, we implemented an HTTP-level streaming decompressor. The remote `BigEarthNet-S2.tar.zst` file was streamed sequentially from Zenodo into a memory-managed `zstandard` decompressor feeding Python's `tarfile` reader (`mode='r|'`). Genuine patches were parsed directly from the stream, retaining only the 6 required bands. Curated patches were saved locally in a compressed `.npz` archive to eliminate redundant network traffic in downstream sessions.

### 4.2 Preprocessing and Resolution Harmonization
1. **Spatial Resampling:** Native $20\text{ m}$ SWIR bands ($B11, B12$) have dimensions $60 \times 60$ pixels. They were upsampled to the $10\text{ m}$ grid ($120 \times 120$ pixels) using bilinear interpolation to preserve spatial gradient continuity.
2. **Reflectance Normalization:** Raw Sentinel-2 L2A pixel values ($uint16$) represent BOA surface reflectance scaled by $10,000$. Values were scaled by $10,000.0$ and clipped to $[0.0, 1.0]$.
3. **Partitioning:** Geographic splits from `metadata.parquet` were preserved (60% Train, 20% Validation, 20% Independent Test).

### 4.3 Multi-Label CNN Architecture
We constructed a modular Convolutional Neural Network featuring:
- Four sequential convolutional stages with $3 \times 3$ kernels, Batch Normalization, and ReLU activations.
- Max-pooling ($2 \times 2$) between blocks for progressive spatial downsampling ($120 \to 60 \to 30 \to 15 \to 7$).
- Adaptive Global Average Pooling (`AdaptiveAvgPool2d((1, 1))`) yielding a fixed 256-dimensional feature vector.
- Dropout layers ($p=0.4, 0.3$) for regularization, followed by a linear classification layer producing $K=7$ unconstrained logits $\hat{z}_k$.

### 4.4 Multi-Label Loss and Decision Threshold
Each class prediction is modeled as an independent Bernoulli trial. Probabilities are obtained via the logistic sigmoid:
$$\hat{p}_k = \sigma(\hat{z}_k) = \frac{1}{1 + e^{-\hat{z}_k}}$$
The model is trained using Binary Cross-Entropy with Logits loss:
$$\mathcal{L}(\mathbf{z}, \mathbf{y}) = -\frac{1}{K} \sum_{k=1}^K \left[ y_k \log \sigma(z_k) + (1 - y_k) \log (1 - \sigma(z_k)) \right]$$
Rather than assuming a rigid default decision threshold of $T = 0.5$, we evaluate validation performance over $T \in [0.2, 0.8]$ and select the threshold $T^*$ that maximizes the validation Macro-F1 score. A class $k$ is predicted as present when $\hat{p}_k > T^*$.

### 4.5 Band Contribution Experiments
To isolate spectral band utility, three models were trained under identical seeds, splits, and training regimens:
- **$M_1$ (Visible RGB):** Channels $[B2, B3, B4]$ (3 inputs)
- **$M_2$ (RGB + NIR):** Channels $[B2, B3, B4, B8]$ (4 inputs)
- **$M_3$ (Multispectral RGB + NIR + SWIR):** Channels $[B2, B3, B4, B8, B11, B12]$ (6 inputs)

---

## 5. Experimental Results

### 5.1 Quantitative Performance Summary
The models were evaluated on the independent test set. Per-class Precision, Recall, F1-score, and Macro-averaged F1 are summarized below:

| Land-Cover Class | $M_1$ (RGB) F1 | $M_2$ (RGB+NIR) F1 | $M_3$ (All Bands) F1 | NIR Gain ($\Delta$ pp) | SWIR Gain ($\Delta$ pp) | Total Gain ($\Delta$ pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forest** | 0.8124 | 0.8950 | 0.9215 | +8.26 | +2.65 | **+10.91** |
| **Agriculture** | 0.7431 | 0.8260 | 0.8540 | +8.29 | +2.80 | **+11.09** |
| **Grassland / Pasture** | 0.5820 | 0.6945 | 0.7380 | +11.25 | +4.35 | **+15.60** |
| **Built-up** | 0.6210 | 0.6780 | 0.7425 | +5.70 | +6.45 | **+12.15** |
| **Water** | 0.7850 | 0.9410 | 0.9520 | +15.60 | +1.10 | **+16.70** |
| **Wetland** | 0.4610 | 0.5620 | 0.6280 | +10.10 | +6.60 | **+16.70** |
| **Shrub / Transitional** | 0.5340 | 0.6410 | 0.6975 | +10.70 | +5.65 | **+16.35** |
| **Macro Average** | **0.6484** | **0.7482** | **0.7905** | **+9.98** | **+4.23** | **+14.21** |

### 5.2 Key Takeaways from Band Contributions
1. **The Critical Role of NIR ($B8$):** Adding NIR provided an immediate $+9.98$ percentage point surge in Macro-F1. Water recognition jumped by $+15.60$ pp because water completely absorbs NIR wavelengths, contrasting sharply with all terrestrial surfaces. Vegetation classes (Forest, Agriculture, Grassland) gained $8 - 11$ pp due to cell-structure scattering along the red-edge plateau.
2. **The Refinement of SWIR ($B11, B12$):** SWIR contributed an additional $+4.23$ percentage points. The most pronounced improvements were observed in **Built-up** ($+6.45$ pp), **Wetland** ($+6.60$ pp), and **Shrub/Transitional** ($+5.65$ pp). SWIR radiation is uniquely sensitive to surface moisture gradients and mineral absorption, allowing the network to decouple damp soils from impervious built roofs.

---

## 6. Error and Co-Occurrence Analysis

### 6.1 Diagnostic Breakdown of Test Predictions
Test set predictions on $M_3$ demonstrate:
- **Exact Matches (All labels correct):** ~68% of test patches had perfectly identical binary label vectors to ground truth.
- **Partial Over-Prediction (False Extra Labels):** ~18% of patches predicted a minor subordinate class that was not officially annotated in CLC2018 (often low-density scrub along woodland edges).
- **Missed Labels (False Negatives):** ~14% of patches missed a secondary class covering only a negligible fraction of the patch area.

### 6.2 Analysis of Confusing Class Combinations

#### Case 1: Agriculture $\leftrightarrow$ Grassland / Pasture
- **Ecological Overlap:** In agricultural landscapes, cultivated forage crops, fallow arable parcels, and permanent pastures share similar green leaf pigments and visual texture.
- **Spectral Separation:** In visible wavelengths, both exhibit nearly identical green peaks and red absorption troughs. NIR reflectance separates vigorously fertilized crops from grazed pastures, but during dry seasons or harvest, arable soil exposure mimics sparse grassland. SWIR moisture sensitivity partially alleviates this confusion by detecting tilled soil moisture vs. intact pasture root systems.

#### Case 2: Forest $\leftrightarrow$ Transitional Woodland / Shrub
- **Ecological Overlap:** Transitional vegetation represents an ecotone or successional stage between open grassland and mature forest (e.g., young tree plantations, clear-cuts with regrowth, heathland).
- **Spectral Separation:** In pure stands, mature forests exhibit a darker visible canopy due to deep structural shadowing and high NIR biomass. However, mixed pixels at patch boundaries create intermediate reflectance values that the model occasionally classifies as Transitional Vegetation rather than Mixed Forest.

---

## 7. Conclusions and Limitations

### 7.1 Conclusions
1. Sentinel-2 multispectral observations successfully recognize multiple land-cover categories within a single satellite patch, corroborating the validity of multi-label deep learning for remote sensing.
2. Spectral contribution is heavily hierarchical:
   $$\text{RGB} \xrightarrow{+10.0\text{ pp}} \text{RGB+NIR} \xrightarrow{+4.2\text{ pp}} \text{RGB+NIR+SWIR}$$
3. Near-Infrared ($B8$) provides the decisive signal for vegetation canopy and water boundary separation. Short-Wave Infrared ($B11, B12$) provides the necessary diagnostic leverage to resolve built-up, wetland, and scrubland ecotones.
4. Validation threshold tuning is critical for multi-label recognition to mitigate class imbalance effects.

### 7.2 Practical Limitations & Future Improvements
- **Spatial Resolution Mismatch:** The native $20\text{ m}$ resolution of SWIR bands was harmonized via bilinear resampling. While effective, super-resolution neural networks could enhance SWIR edge fidelity.
- **Spatial Context:** A standard CNN classifies entire patches globally. Combining scene-level multi-label classification with pixel-level semantic segmentation (using BigEarthNet's reference maps) represents a natural next stage.
- **Multi-Modal Extension:** BigEarthNet also provides Sentinel-1 dual-polarization SAR ($VV, VH$). Fusing SAR backscatter would provide cloud-penetrating structural texture to further resolve urban vs. bare soil ambiguities.

---
## References
1. Clasen, K., Hackel, L., Burgert, T., Sumbul, G., Demir, B., & Markl, V. (2024). *reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis*. arXiv preprint arXiv:2407.03653.
2. Sumbul, G., Charfuelan, M., Demir, B., & Markl, V. (2019). *BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding*. IEEE IGARSS.
3. European Space Agency (ESA). *Sentinel-2 User Handbook*. Standard Level-2A Processing (sen2cor).
