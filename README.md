# SEA-Biofouling-Quantification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Official implementation of the paper:  
**"Automated In Situ Quantification of Offshore Fish Cage Biofouling via Lightweight Semantic Segmentation and Indirect Aperture Geometry Assessment"** (*Submitted to Computers and Electronics in Agriculture*).

---

## 📌 Overview

Biofouling on offshore aquaculture cage netting severely restricts water exchange, compromises dissolved oxygen availability, and increases hydrodynamic drag. Direct pixel-level biofouling segmentation under underwater environments suffers from extreme textural heterogeneity, non-rigid biofouling boundaries, and severe occlusion. 

To overcome these challenges, this study proposes an **indirect aperture-structure biofouling quantification framework**:
1. **Lightweight Semantic Segmentation (`YOLOv11-SEA`)**: Segments unobstructed cage apertures (negative space) rather than irregular biofouling masses. Integrates parameter-free 3D attention (**SimAM**) and an edge-aware **Laplacian of Gaussian (LoG)** loss to preserve fine netting twine contours.
2. **Adaptive Geometrical Scale Matching**: Employs Zhang-Suen topological skeletonization and scale-invariant geometric matching across clean calibration templates to dynamically infer baseline twine coverage ($r_{\text{clean}}$) and safety margin ($\delta$).
3. **Dimensionless Severity Scoring ($S_{\text{norm}}$)**: Quantifies local aperture blockage ratios ($B_k$) with boundary topological inference, yielding a bounded, scale-invariant severity score strictly within $[0, 100\%]$.

---

## 📂 Repository Structure

```text
SEA-Biofouling-Quantification/
├── README.md                 # Project overview and reproduction guide
├── requirements.txt          # Python dependencies
├── demo.py                   # Minimal reproducible pipeline demonstration script
├── models/
│   ├── __init__.py
│   └── yolov11_sea.py        # YOLOv11-SEA architecture & LoG edge loss
├── score/
│   ├── __init__.py
│   ├── scorer.py             # BiofoulingScorer core evaluation engine
│   └── templates/            # Clean netting scale calibration templates
│       ├── binary/           # Clean aperture binary masks (c_01 - c_08)
│       ├── skeleton/         # Extracted 1-pixel netting skeletons
│       └── template_metadata.json
├── samples/                  # De-identified representative test samples
│   ├── sample_01_clean.jpg       # Clean cage mesh sample
│   ├── sample_01_clean_mask.png  # Corresponding aperture mask
│   ├── sample_02_moderate.jpg    # Moderately fouled mesh sample
│   ├── sample_02_moderate_mask.png
│   ├── sample_03_heavy.jpg       # Heavily fouled mesh sample
│   └── sample_03_heavy_mask.png
└── outputs/                  # Directory for generated demonstration overlays
```

---

## ⚙️ Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aetzth/SEA-Biofouling-Quantification.git
   cd SEA-Biofouling-Quantification
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   conda create -n biofouling python=3.9 -y
   conda activate biofouling
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start Demo

Run the end-to-end demonstration script out of the box:

```bash
python demo.py
```

This script performs:
1. **Network Definition Verification**: Instantiates `YOLOv11-SEA` and executes a dummy forward pass to verify multi-scale semantic aggregation.
2. **Dynamic Template Matching**: Computes geometric distance $d(C_j, M_i)$ to find optimal calibration scale and determine dynamic threshold $T = r_{\text{clean}} + \delta$.
3. **Severity Scoring & Visualization**: Evaluates local aperture blockage proportions $B_k$, computes normalized severity score $S_{\text{norm}}$, and exports intuitive visual overlays to `outputs/`.

### Exemplar Evaluation Output

| Sample | Condition | Matched Template | Dynamic Threshold ($T$) | Fouled / Total Apertures | Severity ($S_{\text{norm}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `sample_01_clean.jpg` | Clean | `c_08.png` | 0.4323 | 25 / 46 | **4.74%** |
| `sample_02_moderate.jpg` | Moderate | `c_08.png` | 0.4323 | 20 / 33 | **29.21%** |
| `sample_03_heavy.jpg` | Severe | `c_08.png` | 0.4323 | 15 / 16 | **82.52%** |

---

## 🔒 Data Availability and Governance Policy

- **Minimal Reproducible Kit**:  
  This public repository provides all core algorithmic implementations (`YOLOv11-SEA`, edge-enhanced loss, topological skeletonization, and `BiofoulingScorer`), the clean netting scale calibration templates, and 3 de-identified representative test samples across distinct fouling severity tiers for immediate pipeline verification.

- **Full Cohort and Benchmark Datasets**:  
  The full underwater biofouling dataset (comprising 1,327 high-resolution annotated images captured across diverse illumination, turbidity, and netting deformations) and the physical mussel surrogate validation dataset are proprietary research assets governed by institutional project confidentiality agreements.  
  To protect research intellectual property while supporting academic reproducibility:
  > **The complete annotated image cohort is available for non-commercial academic research upon reasonable request to the corresponding author** (Contact: *[Corresponding Author Name & Email as indicated in the manuscript]*). A formal Data Transfer and Research Use Agreement (DTA) may be required.

---

## 📖 Citation

If you find this repository or methodology helpful in your research, please cite our paper:

```bibtex
@article{biofouling2026sea,
  title={Automated In Situ Quantification of Offshore Fish Cage Biofouling via Lightweight Semantic Segmentation and Indirect Aperture Geometry Assessment},
  author={[Author List as in Manuscript]},
  journal={Computers and Electronics in Agriculture},
  year={2026},
  publisher={Elsevier}
}
```

---

## 📄 License

This repository is distributed under the Apache 2.0 License. See `LICENSE` for details.
