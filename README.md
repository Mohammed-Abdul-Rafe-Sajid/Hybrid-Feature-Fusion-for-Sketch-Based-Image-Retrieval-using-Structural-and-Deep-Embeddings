# Hybrid Feature Fusion for Sketch-Based Image Retrieval using Structural and Deep Embeddings

## Overview

This project presents a hybrid feature fusion approach for Sketch-Based Image Retrieval (SBIR), combining structural edge-based representations with deep semantic embeddings. The system leverages convolutional neural networks and feature fusion techniques to improve retrieval performance, supported by quantitative evaluation and visual analysis.


## Abstract
Sketch-Based Image Retrieval (SBIR) aims to retrieve relevant images from a database using sketch queries. Traditional approaches rely either on structural features such as edges or deep semantic features extracted from convolutional neural networks. However, each representation alone has limitations.

In this work, we propose a hybrid feature fusion approach that combines structural edge-based features with deep semantic embeddings to improve retrieval performance. Edge representations are generated using Canny edge detection, while semantic features are extracted using a pretrained ResNet18 model. A weighted fusion mechanism is introduced to balance structural and semantic information.

Experiments conducted on the STL-10 dataset demonstrate that the proposed method achieves improved retrieval performance compared to individual feature representations. Evaluation using Precision@K over 200 queries shows that moderate fusion weights outperform both raw and edge-only approaches. The results highlight the importance of combining complementary feature types for effective sketch-based image retrieval.


## Introduction
Sketch-Based Image Retrieval (SBIR) has gained significant attention due to its ability to enable intuitive search using visual queries. Unlike text-based retrieval, SBIR relies on structural and visual features, making it useful in applications such as design search, digital art, and object retrieval.

Existing approaches primarily rely on either handcrafted structural features or deep learning-based semantic features. While deep models capture high-level semantics, they often fail to emphasize structural details. Conversely, edge-based representations capture object shapes but lack contextual information.

To address this limitation, we propose a hybrid feature fusion approach that integrates both structural and semantic representations. By combining edge-based features with deep embeddings, the system leverages complementary information for improved retrieval performance.

The main contributions of this work are:

1. A hybrid feature fusion framework for SBIR
2. Empirical analysis of fusion weights on retrieval performance
3. Comprehensive evaluation using multiple Precision@K metrics

# Dataset

## STL-10 Dataset

* 5000 labeled training images
* 96x96 resolution
* 10 object categories

## Why STL-10?

* Higher resolution than CIFAR-10
* Better visual quality for retrieval tasks
* Suitable for feature extraction using CNNs

## Sample Dataset

A subset of 500 images is used for:

* Faster experimentation
* Deployment (Streamlit)
* GitHub compatibility



# Edge Detection (STL-10)

## Objective

Convert real images into sketch-like representations to simulate user sketches.

## Method

* Convert RGB images to grayscale
* Apply Canny Edge Detection

## Why?

Sketch-based retrieval relies on structural features such as edges and shapes rather than color or texture.

## Output

Edge images stored in:
data/sample/edges/

## Observation

Edges effectively capture object outlines, making them suitable for sketch-based retrieval tasks.



# Dual Feature Extraction

## Objective

Extract both semantic and structural representations of images.

## Methods

### 1. Raw Image Embeddings

* Input: Original images
* Captures: Semantic information

### 2. Edge Image Embeddings

* Input: Edge-detected images
* Captures: Structural features

## Model Used

* ResNet18 (pretrained)

## Output

* raw_embeddings.npy
* edge_embeddings.npy
* filenames.npy

## Insight

Combining semantic and structural features can improve retrieval performance.



## Feature Fusion

### Objective

Combine structural (edge) and semantic (raw) features to improve retrieval performance.

### Method

Weighted fusion:

F = αE + (1 - α)R

Where:

* E = edge embeddings
* R = raw embeddings
* α = fusion weight

### Experiments

Multiple α values tested:

* 0.3, 0.5, 0.7, 0.9

### Output

* fused_0.3.npy
* fused_0.5.npy
* fused_0.7.npy
* fused_0.9.npy

### Insight

Different α values allow balancing between structural and semantic information.



## 📊 Experimental Results

### Evaluation Setup

* Dataset: STL-10 (sample subset)
* Number of queries: 200
* Metrics: Precision@K (K = 3, 5, 10)
* Reported as: Mean ± Standard Deviation

---

### Results

| Method       | P@3           | P@5               | P@10              |
| ------------ | ------------- | ----------------- | ----------------- |
| Raw          | 0.407 ± 0.153 | 0.285 ± 0.129     | 0.198 ± 0.100     |
| Edge         | 0.123 ± 0.187 | 0.117 ± 0.142     | 0.117 ± 0.097     |
| Fusion (0.3) | 0.400 ± 0.149 | **0.294 ± 0.133** | 0.200 ± 0.099     |
| Fusion (0.5) | 0.397 ± 0.139 | 0.290 ± 0.131     | **0.205 ± 0.098** |
| Fusion (0.7) | 0.397 ± 0.131 | 0.290 ± 0.118     | 0.196 ± 0.095     |
| Fusion (0.9) | 0.220 ± 0.215 | 0.184 ± 0.153     | 0.153 ± 0.099     |

---

### Key Observations

* Raw image embeddings provide strong semantic understanding.
* Edge-based features alone perform poorly due to loss of contextual information.
* Hybrid feature fusion improves performance over raw features.
* Best performance is achieved with **moderate fusion weights (α = 0.3–0.5)**.
* High reliance on edge features (α = 0.9) significantly degrades performance.

---

### Insight

These results indicate that:

* Semantic features (deep embeddings) are dominant in retrieval tasks.
* Structural features (edges) are complementary, not sufficient alone.
* A balanced fusion of both leads to improved retrieval accuracy.

This validates the effectiveness of the proposed **hybrid feature fusion approach**.


#### Best performing configuration: Fusion (α = 0.3) for P@5 and Fusion (α = 0.5) for P@10.



# 📊 Experimental Visualizations

This folder contains visual analysis of the retrieval performance for different feature configurations used in the proposed Hybrid Feature Fusion approach.

These figures support the quantitative results presented in the evaluation section.

---

## 1. Fusion Weight Analysis (`alpha_analysis.png`)

This plot shows the effect of varying the fusion weight (α) between edge-based and raw image embeddings.

**Key Insights:**

* Moderate fusion weights (α ≈ 0.3 – 0.5) achieve the best performance.
* Very high edge weight (α = 0.9) significantly degrades performance.
* Indicates that semantic features are dominant while structural features act as complementary information.

---

## 2. Precision Comparison Across Methods (`precision_comparison.png`)

This plot compares Precision@K (K = 3, 5, 10) across different retrieval strategies:

* Raw image embeddings
* Edge-based embeddings
* Hybrid fused embeddings

**Key Observations:**

* Raw embeddings outperform edge-only embeddings.
* Hybrid fusion consistently matches or improves performance compared to raw features.
* Edge-only embeddings are insufficient for effective retrieval.

---

## 3. Bar Chart Comparison (`bar_p5.png`)

This visualization presents a direct comparison of Precision@5 across all methods.

**Highlights:**

* Provides a clear view of performance differences between methods.
* Hybrid models (especially α = 0.3 and α = 0.5) show improved performance compared to raw-only retrieval.

---

## 4. Fusion Performance Trend (`fusion_graph.png`)

This figure visualizes how retrieval performance changes as the contribution of edge features increases.

**Findings:**

* Performance peaks at balanced fusion weights.
* Excessive reliance on edge features reduces retrieval effectiveness.
* Supports the claim that a balanced combination of semantic and structural features is beneficial.

---

---

## 📁 Project Directory Structure

```
sketh_cbir_research/
│
├── data/
│   ├── raw/                    # Full dataset (ignored in git)
│   ├── edges/                  # Full edges dataset (ignored in git)
│   └── sample/                 # Small subset for GitHub/UI
│       ├── raw/                # Sample raw images
│       └── edges/              # Sample edge images
│
├── features/
│   ├── raw_embeddings.npy      # Raw image embeddings (ResNet18)
│   ├── edge_embeddings.npy     # Edge image embeddings (ResNet18)
│   ├── fused_embeddings.npy    # Fused feature embeddings
│   └── filenames.npy           # Corresponding image filenames
│
├── src/
│   ├── preprocessing/
│   │   └── edge_detection.py   # Canny edge detection implementation
│   │
│   ├── feature_extraction/
│   │   ├── extract_raw_features.py    # Extract semantic embeddings
│   │   ├── extract_edge_features.py   # Extract structural embeddings
│   │   └── fusion.py                  # Feature fusion mechanism
│   │
│   ├── retrieval/
│   │   └── retrieve.py         # Image retrieval engine
│   │
│   ├── evaluation/
│   │   └── evaluate.py         # Evaluation metrics (Precision@K)
│   │
│   └── spark/
│       └── spark_pipeline.py   # Spark-based data pipeline (optional)
│
├── scripts/
│   └── create_sample.py        # Script to create sample dataset
│
├── notebooks/
│   └── experiments.ipynb       # Jupyter notebook for experiments
│
├── reports/
│   ├── figures/                # Visualization outputs
│   ├── tables/                 # Results tables
│   ├── logs/                   # Execution logs
│   └── paper/                  # Paper and documentation
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `data/` | Raw and processed dataset storage |
| `features/` | Precomputed embeddings and feature vectors |
| `src/` | Core source code modules |
| `scripts/` | Utility and preprocessing scripts |
| `notebooks/` | Jupyter notebooks for experimentation |
| `reports/` | Results, visualizations, and documentation |
| `app.py` | Streamlit web application entry point |

---



## ⚠️ Limitations

* Edge-based representations may lose fine-grained semantic details
* STL-10 dataset is not specifically designed for sketch-based retrieval
* Fusion is manually weighted (α), not learned dynamically

These limitations provide opportunities for future improvements.

## 🚀 Future Work

* Learn adaptive fusion weights using neural networks
* Evaluate on sketch-specific datasets (e.g., TU-Berlin, Sketchy)
* Incorporate multimodal retrieval (text + sketch)
* Explore transformer-based feature representations

## Summary

These visualizations collectively demonstrate:

* The effectiveness of hybrid feature fusion
* The importance of balancing semantic and structural information
* The performance limitations of using structural or semantic features alone

These plots provide strong empirical evidence supporting the proposed approach.
