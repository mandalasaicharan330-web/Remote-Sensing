# Phase 1 Feasibility Report: BigEarthNet-S2 Multi-Label Classification

**Project:** Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification Using the BigEarthNet Dataset  
**Target Platform:** Google Colab (Notebook execution) & Antigravity (Development)  
**Author:** Applied Geophysics Undergraduate Practical  
**Date:** October 2026  

---

## 1. Executive Summary

This feasibility study investigates data-access mechanisms and data integrity for **BigEarthNet Sentinel-2** before committing computational resources or downloading large data files.

> [!IMPORTANT]
> **Key Finding:** We verified that genuine BigEarthNet Sentinel-2 multispectral image patches and their corresponding annotations can be accessed, filtered, and extracted **selectively on the fly without downloading the full ~59 GiB archive**.
>
> In our verification test, **5 genuine image patches** (all 6 required bands: B2, B3, B4, B8, B11, B12) were streamed and decoded directly from the official Zenodo archive using only **0.69 MB** of compressed network transfer in under 8 seconds.

---

## 2. Official Dataset Inspection & Version Details

| Property | Details |
| :--- | :--- |
| **Official Portal** | [bigearth.net](https://bigearth.net/) |
| **Active Dataset Version** | **BigEarthNet v2.0** (also known as **reBEN** — *Refined BigEarthNet*) |
| **Reference Publication** | K. Clasen et al., *"reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis"*, arXiv:2407.03653 (2024) |
| **Repository Host** | Zenodo Record ID: `10891137` (DOI: `10.5281/zenodo.10891137`) |
| **Full Sentinel-2 Archive** | `BigEarthNet-S2.tar.zst` (~58.9 GiB compressed, 63,251,710,377 bytes) |
| **Atmospheric Correction** | Sentinel-2 Level-2A processed via `sen2cor v2.11` (superior to v1.0) |
| **Ground-Truth Reference** | CORINE Land Cover 2018 (`CLC2018 v2020_u1`) |
| **License** | Community Data License Agreement – Permissive (CDLA-Permissive-1.0) |

### Key Improvements in v2.0 over v1.0:
1. **Atmospheric Correction:** Utilizes modern `sen2cor 2.11`, removing calibration artifacts present in earlier releases.
2. **Reduced Label Noise:** Re-annotated against the latest CLC2018 database update, resolving known label noise and misclassifications in v1.0.
3. **Geographical Split Assignment:** Splitting algorithm ensures spatially disjoint tiles between train, validation, and test splits to prevent spatial auto-correlation and data leakage.

---

## 3. Metadata Inspection (`metadata.parquet`)

Before downloading imagery, the official metadata file was retrieved and inspected:
- **File:** `metadata.parquet` (Zenodo record `10891137`)
- **File Size:** **3.44 MB** (downloaded in 13 seconds)
- **Total Valid Patches:** **480,038** (excluding cloud, shadow, and seasonal snow patches)

### 3.1 Metadata Schema & Structure
```text
patch_id:                  string (e.g. 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57')
labels:                    list<string> (CORINE 19-class nomenclature)
split:                     string ('train', 'validation', 'test')
country:                   string (10 European countries)
s1_name:                   string (matching Sentinel-1 SAR patch name)
s2v1_name:                 string (legacy BigEarthNet v1.0 patch name)
contains_seasonal_snow:    bool (all False in recommended metadata)
contains_cloud_or_shadow:  bool (all False in recommended metadata)
```

### 3.2 Dataset Splits & Geographical Distribution
- **Split Distribution:**
  - `train`: 237,871 patches (49.55%)
  - `validation`: 122,342 patches (25.49%)
  - `test`: 119,825 patches (24.96%)
- **Country Distribution:** Finland (155k), Portugal (89.8k), Serbia (73.4k), Lithuania (48.4k), Ireland (48.3k), Austria (43.8k), Belgium (11.2k), Switzerland (4.9k), Luxembourg (3.5k), Kosovo (1.6k).

### 3.3 Multi-Label Characterization
- **Distinct Standard Labels:** 19 classes
- **Multi-Label Frequency:**
  - Only **17.99%** of patches have a single label.
  - **82.01%** of patches are multi-label:
    - 2 labels: 19.72%
    - 3 labels: 28.31%
    - 4 labels: 21.26%
    - 5 labels: 9.44%
    - 6–11 labels: 3.28%
- This confirms the project's central premise: **satellite patches inherently contain multiple land-cover types**.

---

## 4. Verification of Selective Data Access

### 4.1 Access Testing Findings

1. **Direct HTTP Range Requests on Zenodo:**
   - Testing `Range: bytes=0-100` against the 59 GiB Zenodo file returned `HTTP 200 OK` with full content length (`63,251,710,377` bytes).
   - *Conclusion:* Zenodo does not honor partial HTTP range seeking on this archive.
2. **Streaming Decompression (The Solution):**
   - The archive `BigEarthNet-S2.tar.zst` is a Tar stream compressed with Zstandard (`.zst`).
   - By initiating an HTTP stream and piping incoming bytes through a streaming decompressor (`zstandard.ZstdDecompressor().decompressobj()`) into a sequential tar reader (`tarfile.open(mode='r|')`), patches can be read sequentially and extracted **directly from the front of the archive**.
   - As soon as the desired number of patches is retrieved, the connection is closed immediately, downloading only the exact bytes required.

### 4.2 Verified Small-Sample Test Results

We executed the streaming extraction script to retrieve **5 complete genuine Sentinel-2 patches** retaining only the required 6 bands:

| Band | Spectral Region | Native Res. | Shape | Dtype | Reflectance Range | Mean Value |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B02** | Blue (490 nm) | 10 m | $120 \times 120$ | `uint16` | 80 – 1,736 | 341.6 |
| **B03** | Green (560 nm) | 10 m | $120 \times 120$ | `uint16` | 190 – 2,060 | 638.6 |
| **B04** | Red (665 nm) | 10 m | $120 \times 120$ | `uint16` | 105 – 2,322 | 477.6 |
| **B08** | NIR (842 nm) | 10 m | $120 \times 120$ | `uint16` | 1,021 – 6,808 | 3,610.6 |
| **B11** | SWIR-1 (1610 nm) | 20 m | $60 \times 60$ | `uint16` | 818 – 3,401 | 1,951.6 |
| **B12** | SWIR-2 (2190 nm) | 20 m | $60 \times 60$ | `uint16` | 323 – 3,044 | 1,070.1 |

**Data transfer for all 5 patches (30 GeoTIFF files):** **0.69 MB** (approx. 138 KB per full multispectral patch).

### 4.3 Ground-Truth Cross-Check with Official Metadata

The patch IDs extracted from the archive were matched against `metadata.parquet`:
- `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57`:
  - Country: Austria | Split: test
  - Labels: `['Arable land', 'Broad-leaved forest', 'Mixed forest', 'Pastures']`
- `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_27_55`:
  - Country: Austria | Split: test
  - Labels: `['Arable land', 'Broad-leaved forest', 'Inland waters', 'Mixed forest']`
- `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_27_56`:
  - Country: Austria | Split: test
  - Labels: `['Arable land', 'Broad-leaved forest', 'Coniferous forest', 'Inland waters', 'Mixed forest']`
- `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_27_57`:
  - Country: Austria | Split: test
  - Labels: `['Broad-leaved forest', 'Complex cultivation patterns', 'Coniferous forest', 'Inland waters', 'Mixed forest']`
- `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_27_58`:
  - Country: Austria | Split: test
  - Labels: `['Broad-leaved forest', 'Complex cultivation patterns', 'Inland waters', 'Land principally occupied by agriculture, with significant areas of natural vegetation']`

---

## 5. Visual Composites of Retrieved Sample

To verify image quality, true-colour ($B4$-$B3$-$B2$) and standard false-colour infrared ($B8$-$B4$-$B3$) composites were produced directly from the extracted GeoTIFF files:

![True Colour and False Colour Infrared Composites of Sample Patches](file:///C:/Users/manda/.gemini/antigravity-ide/brain/71c5e37b-edd5-40e1-bd63-a6ad71391d42/test_composites.png)

### Remote Sensing Observations:
- **Vegetation Discrimination:** In True Colour, dense broad-leaved/mixed forests appear dark green, while crop fields exhibit yellowish/light-green tones. In Color Infrared (FCC), healthy vegetative canopy reflects strongly in NIR ($B8$) appearing bright crimson red.
- **Water Bodies:** In patch `3UUP_27_55`, the river in the bottom-right corner absorbs NIR radiation strongly and appears dark blue/black in FCC, offering sharp separation from surrounding pastures and woodland.
- **Resolution Characteristics:** Visible & NIR bands ($10\text{ m}$) capture fine field boundaries and roads, whereas SWIR bands ($20\text{ m}$) supply critical moisture and soil differentiation.

---

## 6. Proposed Resource & Scaling Plan for Phase 2

| Subset Size | Estimated Download | Extraction Time (Colab) | Cache Footprint (`.npz`) | Recommended Use |
| :---: | :---: | :---: | :---: | :---: |
| **100 patches** | ~14 MB | ~10 – 15 seconds | ~8 MB | Pipeline validation & debugging |
| **1,000 patches** | ~140 MB | ~1 – 2 minutes | ~80 MB | Fast experimentation & testing |
| **2,500 patches** | ~350 MB | ~3 – 5 minutes | ~200 MB | Full robust training & benchmark |

> [!TIP]
> **Colab Session Strategy:**
> 1. Use the tested streaming decompressor in Google Colab to download a curated set of 1,000–2,500 patches without ever touching the 59 GiB archive.
> 2. Save the curated patches into Google Drive or Colab storage as a single compressed file (`bigearthnet_subset.npz`) containing resampled 6-band arrays and multi-hot labels.
> 3. Future notebook runs reload this local cache instantly in $<2$ seconds.

---

## 7. Next Steps (Awaiting Approval)

Phase 1 is complete and verified with genuine data. Upon your review and approval, we will proceed to **Phase 2 (Curating the subset)**:
1. Stream ~1,000–2,000 genuine patches balanced across splits and classes.
2. Resample 20 m SWIR bands ($B11, B12$) to $120 \times 120$ to match the 10 m working grid.
3. Map 19 CORINE classes to the 7 project broad classes (*Forest, Agriculture, Grassland/Pasture, Built-up, Water, Wetland, Shrub/Transitional*).
4. Save the verified dataset ready for Colab notebook development in Phase 3.
