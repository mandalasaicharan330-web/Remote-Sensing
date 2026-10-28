# Oral Presentation Speaker Notes: Multi-Label Land-Cover Recognition Using Sentinel-2

**Project:** Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification  
**Target Duration:** ~8–10 Minutes (30–60 seconds per slide across 10 slides)  
**Presenter:** Undergraduate Student in Applied Geophysics  
**Companion Deck:** `presentation/final_project_presentation.pptx`

---

### Slide 1: Title & Introduction
- **Target Timing:** 30–45 seconds
- **Visual:** Clean dark navy title slide with project title, author, and date.
- **Spoken Script:**
  > "Good morning, Professor and colleagues. Today I am presenting our Applied Remote Sensing project on **Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification**. 
  > In this project, we investigated how satellite imagery can identify multiple co-occurring surface types in complex landscapes, and we tested how adding Near-Infrared and Shortwave-Infrared bands influences machine learning performance compared to standard visible optical photographs.
  > All data and experiments were conducted on genuine Sentinel-2 Level-2A imagery from the BigEarthNet v2.0 benchmark."

---

### Slide 2: Background & Motivation: The Mixed Landscape Challenge
- **Target Timing:** 45–60 seconds
- **Visual:** Side-by-side comparison between the conventional single-label dilemma and the multi-label paradigm.
- **Spoken Script:**
  > "To understand why this work is necessary, look at how satellite imagery is traditionally classified. Conventional remote sensing forces every image pixel or patch into a single, mutually exclusive category—for example, calling an entire square kilometer 'Forest.'
  > But real landscapes are complex mosaics. A single 1.2-kilometer patch typically contains agricultural fields bordering woodlots, small rivers, and rural homes. If we force a single label, we discard secondary classes like freshwater streams or grazing pastures, which are critical for hydrology and environmental monitoring.
  > In our dataset, we found that **94.4% of all satellite patches contain two or more co-occurring land covers**, averaging 3.00 classes per patch. Multi-label classification solves this by modeling each land-cover type as an independent binary question: is this class present in the patch, yes or no?"

---

### Slide 3: Research Questions & Project Objectives
- **Target Timing:** 45 seconds
- **Visual:** Three structured cards detailing the research question, band ablation protocol, and evaluation strategy.
- **Spoken Script:**
  > "Our primary research question was: **Can multispectral Sentinel-2 observations accurately resolve multiple co-occurring surface types, and how does expanding spectral dimensionality impact deep learning on a small curated dataset?**
  > To answer this, we designed three core objectives:
  > First, curate a certified, leakage-free dataset from BigEarthNet v2.0.
  > Second, conduct a strictly controlled **Band Ablation Study** comparing three input models under identical conditions:
  > M1 using visible RGB, M2 adding Near-Infrared, and M3 adding Shortwave-Infrared.
  > And third, evaluate multi-label metrics rigorously, diagnosing error patterns, minority class limitations, and decision thresholds."

---

### Slide 4: Sentinel-2 Spectral Bands & BigEarthNet v2.0 Dataset
- **Target Timing:** 50–60 seconds
- **Visual:** Left column listing the 6 diagnostic bands; right column listing dataset provenance and split counts.
- **Spoken Script:**
  > "Here are the data and spectral bands we used. Sentinel-2 carries the MultiSpectral Instrument. We selected six key diagnostic bands:
  > In the visible spectrum at 10-meter resolution: Blue B2, Green B3, and Red B4, which capture chlorophyll absorption.
  > In the near-infrared at 10 meters: Band B8 at 842 nanometers, which reflects strongly off the spongy mesophyll cells inside healthy leaves, but is completely absorbed by liquid water.
  > And in the shortwave-infrared: Bands B11 and B12 at 20-meter resolution, which are sensitive to leaf water content, soil moisture, and artificial surfaces.
  > Our dataset consists of 500 genuine patches from Tile `T33UUP` in the Austrian Alps, strictly divided into the official splits: 300 for training, 100 for validation, and 100 for independent testing. We verified strict patch-level disjointness between these partitions, while recognizing that all patches originate from the same regional alpine tile."

---

### Slide 5: Reflectance Preprocessing & Multispectral Views
- **Target Timing:** 45–60 seconds
- **Visual:** Pipeline description on the left; True-Colour (B4-B3-B2) and Color-Infrared (B8-B4-B3) 3×3 patch grids on the right.
- **Spoken Script:**
  > "Before training, we applied two essential physical preprocessing steps.
  > First, because bands B11 and B12 have a native 20-meter resolution, we resampled them to 10 meters—120 by 120 pixels—using bilinear interpolation. Bilinear interpolation computes smooth weighted spatial gradients without creating blocky stair-step artifacts.
  > Second, we divided raw digital numbers by 10,000 to scale values to true physical Bottom-Of-Atmosphere surface reflectance between 0.0 and 1.0.
  > On the right, you can see the diagnostic power of multispectral imaging. The top figure shows natural True-Colour RGB. The bottom figure shows Color-Infrared, where Band 8 is displayed as red. You can immediately see how healthy forest canopy and pastures pop out in brilliant scarlet red, while water bodies appear jet black. This physical contrast is what the neural network learns to exploit."

---

### Slide 6: Seven Target Classes & Label Distribution
- **Target Timing:** 45–50 seconds
- **Visual:** Comprehensive data table showing counts and percentages of all 7 broad classes across Train, Validation, and Test.
- **Spoken Script:**
  > "We mapped the 19 detailed CORINE Land Cover classes into seven broad, ecologically meaningful categories, shown here with their split distributions.
  > Forest, Agriculture, and Grassland are the dominant landscape types in this alpine tile, appearing in 50% to 87% of all patches.
  > Built-up areas account for about 44%, and Water bodies represent 18%.
  > However, notice the bottom two rows. Shrub and transitional vegetation appears in 23% of the dataset.
  > Most importantly, **Wetland is extremely rare in this alpine region**. In the official training split, there were only **8 positive Wetland patches out of 300**—just 2.7%. Please keep this number in mind, because it explains our most important error analysis finding later."

---

### Slide 7: CNN Architecture & Multi-Label Methodology
- **Target Timing:** 45–60 seconds
- **Visual:** Architecture overview on the left; optimization protocol and threshold calibration on the right.
- **Spoken Script:**
  > "Now turning to our machine learning pipeline. We designed a compact convolutional neural network called `MultispectralCNN` with approximately 617,000 trainable parameters. It features four convolutional blocks with Batch Normalization and ReLU, followed by global average pooling and dropout to prevent overfitting on our 300 training patches.
  > Because this is a multi-label problem, the output layer does not use a softmax. Instead, it outputs seven independent logits, which pass through a sigmoid activation to produce separate probabilities between 0 and 1.
  > We optimized the network using `BCEWithLogitsLoss` and the AdamW optimizer with cosine learning rate scheduling and early stopping based on validation loss.
  > We also calibrated decision thresholds strictly on the validation set, ensuring that test labels were never accessed during training or hyperparameter selection."

---

### Slide 8: Experimental Results: M1 vs M2 vs M3 Band Ablation
- **Target Timing:** 60 seconds
- **Visual:** Performance summary table on the left; grouped bar chart of Macro-F1, Micro-F1, and Exact Match on the right.
- **Spoken Script:**
  > "Here are our central experimental results on the independent 100-patch test set.
  > Looking at the standard 0.5 threshold:
  > Baseline M1, using visible RGB alone, achieved a Macro-F1 of **0.5978** and an Exact Match ratio of **12%**.
  > **Model M2, which adds the Near-Infrared band B8, was our clear winner.** Macro-F1 jumped by over 11 percentage points to **0.7110**, Micro-F1 reached **0.8205**, and the Exact Match ratio more than tripled to **38.0%**, outperforming baseline M1 on the observed test metrics.
  > But look at Model M3, which added the SWIR bands B11 and B12. Macro-F1 dropped back to **0.5673**, and Exact Match fell to **5%**.
  > When we used validation-tuned thresholds, M1 reached 0.7076, M2 achieved 0.7020, and M3 reached 0.6559.
  > The key takeaway is clear: adding NIR was strongly beneficial, while adding SWIR under these experimental conditions underperformed M2."

---

### Slide 9: Per-Class Performance, Error Analysis & Wetland Failure
- **Target Timing:** 60 seconds
- **Visual:** Grouped per-class F1 bar chart on the right; key error bullet points on the left.
- **Spoken Script:**
  > "Let's examine why M2 succeeded and why M3 struggled by looking at the per-class results.
  > Forest was recognized almost perfectly by all models, with F1 scores above 0.96.
  > The biggest breakthrough with NIR was in **Shrub and Transitional Vegetation**. In visible RGB, scrub looks like dark green trees, giving an F1 of only 0.35. Adding NIR caused Shrub F1 to surge to **0.9114** because the lower scrub canopy scatters NIR differently than dense mature forest. Water F1 also jumped to **0.8947** with 100% precision.
  > Why did M3 fail? In M3, the network suffered severe false alarms on Grassland—it generated 57 false positives, predicting pasture in 95 out of 100 test patches, which collapsed precision to 40%.
  > Finally, look at Wetland: **all three models scored exactly zero F1 on Wetland at threshold 0.5**. Because the training set contained only 8 Wetland examples out of 300, the unweighted cross-entropy loss penalized false alarms so heavily that the model suppressed predicted probabilities below 0.5 for all wetland patches."

---

### Slide 10: Conclusions, Limitations & Future Directions
- **Target Timing:** 45–60 seconds
- **Visual:** Three clean summary cards: Conclusions, Limitations, and Future Work.
- **Spoken Script:**
  > "To conclude our presentation:
  > First, multi-label classification is essential for remote sensing because 94.4% of real landscape patches in our dataset contain mixed surface types.
  > Second, Near-Infrared is the single most valuable spectral band for terrestrial classification, substantially improving vegetation differentiation and water boundary recognition.
  > Third, M2 performed best on this curated test set at the default threshold of 0.5. M3 underperformed M2, achieving a Macro-F1 of 0.5673 compared with 0.7110 for M2. Small training-sample size, spectral collinearity, resampling effects, and optimization instability are possible explanations, but their individual contributions were not independently tested.
  > Our main limitations are that all 500 patches come from one alpine tile in Austria, so these results cannot be generalized globally, and Wetland suffered from severe training imbalance.
  > For future work, we recommend implementing positive-class loss weighting or Focal Loss to help detect rare wetlands, and using transfer learning from continental-scale pre-trained models.
  > Thank you very much for your time. I would be happy to answer any questions."

---

### Defense Q&A Preparation: Anticipated Questions & Answers

1. **Question:** *Why did M2 outperform M3 if M3 has more spectral information?*  
   **Answer:** "In remote sensing, more bands mean higher dimensionality. M3 added two 20-meter SWIR bands, expanding the input from 4 to 6 channels. With only 300 training patches, the network had too many degrees of freedom relative to the sample size, causing it to overfit early (Epoch 3) and over-predict pasture. On large datasets with 100,000 patches, SWIR typically helps, but on small datasets, M2 strikes the best bias-variance balance."

2. **Question:** *Why did you get 0.0000 F1 on Wetland, and how would you fix it?*  
   **Answer:** "In tile T33UUP, there were only 8 positive Wetland patches in the training split—2.7% of the data. Because standard BCE loss treats all mistakes equally, the model learned that predicting zero for Wetland avoided penalties on the 292 negative patches. To fix this, we would use positive-class loss weighting, where misclassifying a rare wetland carries a much higher penalty, or use focal loss."

3. **Question:** *Did you use test labels when calibrating your decision thresholds?*  
   **Answer:** "No, absolutely not. All threshold grid searching was performed strictly on the 100-patch validation set. Test labels were only evaluated once as an unbiased final benchmark."
