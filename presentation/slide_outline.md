# Presentation Outline: Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification

**Project:** Remote Sensing Practical Project 6 (BigEarthNet-S2)  
**Target Duration:** 10–12 Minutes (8–10 Slides)  
**Presenter:** Applied Geophysics Undergraduate Student  

---

## Slide 1: Title & Problem Statement
- **Title:** Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification
- **Subtitle:** Investigating Co-Occurring Surface Types with BigEarthNet v2.0
- **The Core Problem:**
  - Real landscapes are heterogeneous: agricultural fields border woodlots; rivers wind past settlements.
  - Conventional single-label classification forces an image patch into one class, discarding secondary classes and failing on mixed pixels.
- **The Solution:**
  - **Multi-Label Classification:** A patch can simultaneously carry multiple labels:
    $$\mathbf{y} = [y_1, y_2, \dots, y_K], \quad y_k \in \{0, 1\}$$
- **Research Question:**
  - *Can visible, NIR, and SWIR observations accurately identify multiple co-occurring surface types, and which classes remain difficult to separate?*

---

## Slide 2: Sentinel-2 Spectral Bands: Visible, NIR, and SWIR
- **Sensor Overview:** ESA Sentinel-2 MSI (MultiSpectral Instrument)
- **The 6 Retained Diagnostic Bands:**
  - **Visible Spectrum ($10\text{ m}$):**
    - $B2$ (Blue, $490\text{ nm}$): Atmospheric penetration, water penetration.
    - $B3$ (Green, $560\text{ nm}$): Chlorophyll green peak, turbidity.
    - $B4$ (Red, $665\text{ nm}$): Maximum chlorophyll absorption.
  - **Near-Infrared ($10\text{ m}$):**
    - $B8$ (NIR, $842\text{ nm}$): Internal leaf spongy mesophyll scattering (vegetation plateau); total absorption by pure water.
  - **Short-Wave Infrared ($20\text{ m}$):**
    - $B11$ (SWIR-1, $1610\text{ nm}$): Foliar water absorption, soil moisture, built-up surfaces.
    - $B12$ (SWIR-2, $2190\text{ nm}$): Dry biomass, mineral/soil response.
- **Data Representation:** Multispectral input cube $\mathbf{X} = [B2, B3, B4, B8, B11, B12] \in \mathbb{R}^{6 \times 120 \times 120}$.

---

## Slide 3: BigEarthNet v2.0 & Simplified Class Scheme
- **Dataset Provenance:**
  - BigEarthNet v2.0 (*reBEN*, TU Berlin, 2024).
  - Atmospherically corrected with `sen2cor v2.11` (BOA reflectance).
  - Ground truth derived from CORINE Land Cover 2018 (`CLC2018 v2020_u1`).
  - Spatial-leakage-free geographical split assignment across European regions.
- **Simplified 7-Class Scheme:**
  1. **Forest:** Broad-leaved, coniferous, and mixed forest.
  2. **Agriculture:** Arable land, permanent crops, complex cultivation, agro-forestry.
  3. **Grassland / Pasture:** Managed pastures, natural grasslands.
  4. **Built-up:** Urban fabric, industrial/commercial infrastructure.
  5. **Water:** Inland lakes/rivers, marine coastal waters.
  6. **Wetland:** Inland marshes, peatbogs, coastal salt marshes.
  7. **Shrub / Transitional:** Transitional woodland, scrub, moors, heathland.
- **Data Access Feasibility:**
  - Avoided 59 GiB download by implementing an on-the-fly HTTP streaming decompressor.

---

## Slide 4: Multispectral Visualizations: RGB & False-Colour Composites
- **Three Diagnostic Views of the Same Scene:**
  - **True Colour ($B4-B3-B2$):** Natural human perception; subtle differences between crop tones.
  - **Color-Infrared (CIR: $B8-B4-B3$):** Healthy vegetation canopy reflects intensely in NIR $\to$ bright red. Water absorbs NIR $\to$ jet black boundary.
  - **SWIR Composite ($B12-B8-B4$):** Differentiates soil moisture, urban fabric, and dry biomass.
- **Mean Spectral Signatures ($\bar{\mathbf{s}}_k$):**
  - Vegetation shows characteristic "red-edge" jump from $B4$ to $B8$.
  - Water shows steep drop towards zero in NIR and SWIR.
  - Built-up exhibits monotonic, flatter reflectance curves.

---

## Slide 5: Label Frequencies & Co-Occurrence Matrix
- **Key Empirical Finding:**
  - **82.01% of image patches are multi-label!** (Only 18% contain a single land-cover type).
  - Modal distribution: 3 co-occurring labels per patch ($28.3\%$).
- **Co-Occurrence Matrix ($C_{ij}$ Highlights):**
  - **Agriculture + Grassland:** Most frequent pairing in European farming mosaics.
  - **Forest + Shrub/Transitional:** Common along clearings, foothills, and forest boundaries.
  - **Agriculture + Built-up:** Characteristic of peri-urban and village field fringes.

---

## Slide 6: Multi-Label Classification Methodology
- **Deep Learning Architecture:**
  - Convolutional Neural Network (CNN) tailored for multispectral patches.
  - 4 convolutional blocks with Batch Normalization + ReLU + Max Pooling.
  - Adaptive Global Average Pooling + Dropout ($p=0.4$) to prevent overfitting.
- **Multi-Label Formulation:**
  - Sigmoid output activation: $\hat{p}_k = \sigma(\hat{z}_k) \in [0, 1]$.
  - Loss function: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`).
- **Validation Threshold Tuning:**
  - Selected optimal threshold $T^*$ over validation grid $[0.2, 0.8]$ maximizing Macro-F1.
  - Prevents under-predicting minority classes caused by standard $0.5$ cutoff.

---

## Slide 7: Band Contribution Experiment & Comparison
- **Controlled Experiment:** Identical architecture, seeds, and splits across 3 configurations:
  - $M_1$: Visible RGB ($B2, B3, B4$)
  - $M_2$: RGB + NIR ($B2, B3, B4, B8$)
  - $M_3$: Multispectral ($B2, B3, B4, B8, B11, B12$)
- **Performance Results (F1-Scores):**
  - **Macro-F1:** $M_1 = 0.648 \xrightarrow{+10.0\text{ pp}} M_2 = 0.748 \xrightarrow{+4.2\text{ pp}} M_3 = 0.791$ (Total Gain: $+14.2\text{ pp}$).
  - **Water:** Jumps $+15.6\text{ pp}$ upon adding NIR (NIR absorption signature).
  - **Built-up & Wetland:** Gain $+6.5\text{ pp}$ and $+6.6\text{ pp}$ upon adding SWIR (moisture & surface texture sensitivity).

---

## Slide 8: Error Analysis & Difficult Class Pairs
- **Prediction Analysis on Test Set:**
  - Exact match on all labels: ~68%.
  - Dominant error: Secondary subordinate class missed or minor false alarm at parcel borders.
- **Two Major Confusions Identified:**
  1. **Agriculture $\leftrightarrow$ Grassland/Pasture:**
     - Both share herbaceous green chlorophyll responses.
     - SWIR helps separate soil moisture in plowed fields from permanent pasture roots.
  2. **Forest $\leftrightarrow$ Transitional Woodland/Shrub:**
     - Scrub represents an ecotone (intermediate biomass density) between open land and mature canopy.
     - Mixed pixels along canopy edges create borderline sigmoid probabilities.

---

## Slide 9: Conclusions & Geophysical Takeaways
- **Answers to Research Questions:**
  1. *Sentinel-2 observations successfully resolve multiple surface types in a single patch.*
  2. *Infrared bands provide non-redundant, physically critical information:*
     - **NIR ($B8$)** provides the primary leap in vegetation canopy and water recognition.
     - **SWIR ($B11, B12$)** provides subtle discrimination for built-up, wetland, and moisture-variable classes.
- **Practical Workflow Insight:**
  - On-the-fly streaming extraction makes working with massive Earth observation archives feasible on standard student environments like Google Colab without 59 GiB local downloads.

---

## Slide 10: Acknowledgments & Q&A
- **Dataset:** BigEarthNet v2.0 (reBEN), TU Berlin RSiM & DIMA.
- **Software Stack:** PyTorch, PyArrow, Pillow, Matplotlib, Scikit-learn, Google Colab.
- **Thank you! Questions and Discussion.**
