# Phase 2 Technical Report: Dataset Curation and Preprocessing

**Project:** Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification  
**Phase:** Phase 2 (Dataset Curation and Preprocessing)  
**Date:** October 2026  
**Curated Dataset Path:** `data/curated_dataset_phase2.npz`  

---

## 1. Executive Summary

Phase 2 successfully establishes the preprocessing pipeline, validates the 7-class label mapping, and curates a verified dataset of **100 genuine Sentinel-2 patches** spanning the official European geographic splits:
- **Train Set:** 60 genuine patches (60.0%)
- **Validation Set:** 20 genuine patches (20.0%)
- **Test Set:** 20 genuine patches (20.0%)

Strict isolation checks confirm **zero patch ID overlap** across splits:
- Train ∩ Validation = ∅
- Train ∩ Test = ∅
- Validation ∩ Test = ∅

---

## 2. Preprocessing & Radiometric Scaling

### 2.1 Band Ordering & Geometry
The 6 required Sentinel-2 bands are processed in canonical spectral order:
1. `B02`: Blue (490 nm, native 10 m, 120 x 120)
2. `B03`: Green (560 nm, native 10 m, 120 x 120)
3. `B04`: Red (665 nm, native 10 m, 120 x 120)
4. `B08`: Near-Infrared (842 nm, native 10 m, 120 x 120)
5. `B11`: SWIR-1 (1610 nm, native 20 m, 60 x 60 -> resampled to 120 x 120)
6. `B12`: SWIR-2 (2190 nm, native 20 m, 60 x 60 -> resampled to 120 x 120)

### 2.2 Spatial Harmonization via Bilinear Resampling
To form a uniform multispectral tensor X in R^(6 x 120 x 120), native 20 m SWIR bands (`B11`, `B12`) are resampled from 60 x 60 to 120 x 120 using **bilinear interpolation**. Bilinear interpolation preserves spatial reflectance continuity across field and canopy boundaries without introducing step artifacts. Original raw GeoTIFF files remain completely unmodified in `data/sample_patches/` and `data/curated_patches/`.

### 2.3 Radiometric Scaling Transformation
Raw digital numbers (DN) represent Bottom-Of-Atmosphere (BOA) surface reflectance scaled by 10,000 (sen2cor v2.11). The transformation applied is:
$$\rho_b(x, y) = \text{clip}\left(\frac{\text{DN}_b(x, y)}{10000.0}, 0.0, 1.0\right)$$
- **Rationale for Division by 10,000:** Matches the official ESA Sentinel-2 L2A specification to recover physical surface reflectance (0.0 = 0%, 1.0 = 100%).
- **Rationale for Clipping to [0, 1]:** Mitigates extreme specular glint and cloud edge saturation (>10,000) while suppressing slight negative sensor calibration noise (<0).

---

## 3. Seven-Class Label Mapping Scheme

All 19 official CORINE labels from BigEarthNet v2.0 were mapped into the 7 broad classes:
1. **Forest:** Broad-leaved forest, Coniferous forest, Mixed forest
2. **Agriculture:** Arable land, Permanent crops, Complex cultivation patterns, Land principally occupied by agriculture with natural vegetation, Agro-forestry areas
3. **Grassland/Pasture:** Pastures, Natural grassland and sparsely vegetated areas
4. **Built-up/Artificial Surfaces:** Urban fabric, Industrial or commercial units
5. **Water:** Inland waters, Marine waters
6. **Wetland:** Inland wetlands, Coastal wetlands
7. **Shrub/Transitional Vegetation:** Transitional woodland/shrub, Moors, heathland and sclerophyllous vegetation
- **Documented Unmapped Category:** *Beaches, dunes, sands* represents bare coastal mineral surfaces (<0.28% of dataset). It is excluded from the 7 target classes to prevent arbitrary label contamination.

Every patch is encoded as a 7-element binary multi-hot target vector y in {0, 1}^7.

---

## 4. Curated Sample Class Frequencies & Multi-Label Statistics

| Broad Class | Total Count | Train (60) | Val (20) | Test (20) | Prevalence (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Forest | 86 | 46 | 20 | 20 | 86.0% |
| Agriculture | 64 | 34 | 15 | 15 | 64.0% |
| Grassland/Pasture | 82 | 51 | 19 | 12 | 82.0% |
| Built-up/Artificial Surfaces | 69 | 47 | 15 | 7 | 69.0% |
| Water | 43 | 20 | 10 | 13 | 43.0% |
| Wetland | 2 | 0 | 2 | 0 | 2.0% |
| Shrub/Transitional Vegetation | 0 | 0 | 0 | 0 | 0.0% |

- **Multi-Label Distribution:**
  - 1 label: 3 patches (3.0%)
  - 2 labels: 13 patches (13.0%)
  - 3 labels: 36 patches (36.0%)
  - 4 labels: 31 patches (31.0%)
  - 5 labels: 17 patches (17.0%)

---

## 5. Generated Artifacts & Verification Outputs

- **Curated Dataset Archive:** `data/curated_dataset_phase2.npz` (N=100)
- **Patch Level Inventory Table:** `outputs/metrics/phase2_curated_patches.csv`
- **Class Frequencies Table:** `outputs/metrics/phase2_class_frequencies.csv`
- **Band Validation Audit:** `outputs/metrics/phase2_band_validation.csv`
- **True-Colour Composite Figure:** `outputs/figures/phase2_true_colour.png`
- **False-Colour (CIR & SWIR) Composite Figure:** `outputs/figures/phase2_false_colour.png`
