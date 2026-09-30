# Early Parkinson's Disease Detection & Grad-CAM Visual Explainability

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision and explainable AI (XAI) system for early-stage Parkinson's Disease screening using hand-drawn spiral, wave, and meander stroke patterns. Built with PyTorch ResNet-18 transfer learning, modern Albumentations data augmentation, Grad-CAM feature attention mapping, automated PDF report generation, and an interactive Streamlit point-of-care web application.

---

## 🌟 Key Features & Differentiators

- **Dual-Source Dataset Pipeline:** Integrates Kaggle Parkinson's Drawings dataset and HandPD (NIATS-UFU) meander & spiral datasets (1,144 total samples).
- **Standardized Preprocessing Engine (`src/preprocessing.py`):** Grayscale conversion, Gaussian noise filtering ($3\times3$), adaptive thresholding with Otsu fallback, and stroke morphological extraction.
- **Direction-Preserving Data Augmentation:** Rotations ($\pm 15^\circ$), affine scaling ($0.9-1.1\times$), elastic/grid distortion, and Gaussian noise. **Explicitly excludes flips** to preserve clockwise vs. counter-clockwise stroke direction semantics.
- **ResNet-18 Transfer Learning Backbone:** Fast point-of-care inference speed utilizing a two-phase fine-tuning strategy (freezing early blocks followed by unfreezing `layer4` residual blocks).
- **Grad-CAM Explainability (`src/explainability.py`):** Generates $224\times224$ heatmap overlays on stroke regions targeting `layer4[-1]` to provide clinicians with clear visual rationale behind model predictions.
- **Automated PDF Diagnostic Report Export (`src/report.py`):** Downloadable clinical summaries containing preprocessed images, Grad-CAM overlays, prediction confidence %, stroke smoothness metrics, timestamps, and disclaimers.
- **Streamlit Point-of-Care Web App (`app.py`):** In-browser upload, sample testing, live visual heatmap visualization, and one-click PDF export.

---

## 📊 Empirical Performance & Benchmark Results

The model was evaluated using a stratified split and 5-fold cross-validation:

| Metric | Score | Clinical Relevance |
| :--- | :--- | :--- |
| **Accuracy** | **88.21%** | Overall classification accuracy across held-out test data |
| **Precision** | **87.50%** | Minimizes false positive diagnoses |
| **Recall (Sensitivity)** | **96.86%** | **Critical for clinical screening** (minimizes missed Parkinson's cases) |
| **F1-Score** | **0.9194** | Harmonic mean balancing precision and recall |
| **ROC-AUC** | **0.9429** | High class discrimination performance |

```
Classification Report:
               precision    recall  f1-score   support

     Healthy       0.91      0.69      0.78        70
 Parkinson's       0.88      0.97      0.92       159

    accuracy                           0.88       229
```

---

## 🏗 System Architecture & End-to-End Flow

```
[Input Image] 
      │
      ▼
[Preprocessing Engine] ──► (Grayscale ➔ Denoise ➔ Adaptive Threshold ➔ Morphology ➔ 224x224 Norm)
      │
      ▼
[ResNet-18 Backbone]   ──► (ImageNet Pretrained ➔ Phase 1 Head ➔ Phase 2 Layer4 Fine-tuning)
      │
      ├───────────────────────────────┐
      ▼                               ▼
[Binary Classification]      [Grad-CAM Heatmap Engine]
(Healthy vs. Parkinson's)    (layer4[-1] Activation Map)
      │                               │
      └───────────────┬───────────────┘
                      ▼
        [Post-processing & PDF Report Engine]
                      │
                      ▼
           [Streamlit Web GUI Output]
```

---

## 📁 Repository Structure

```
.
├── dataset/                    # Dual-source raw image directories
│   ├── kaggle/                 # Kaggle spiral & wave drawings
│   ├── Spiral_HandPD/          # NIATS-UFU HandPD spiral drawings
│   └── Meander_HandPD/         # NIATS-UFU HandPD meander drawings
├── data/
│   └── metadata.csv            # Unified metadata index (1,144 records)
├── models/
│   └── model.pt                # Fine-tuned ResNet-18 weights (44.7 MB)
├── results/                    # Saved training curves, confusion matrix & reports
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── classification_report.txt
├── scripts/
│   └── inspect_datasets.py     # Dataset indexing & CSV generator
├── src/
│   ├── __init__.py
│   ├── augmentation.py         # Albumentations pipeline (no flips)
│   ├── dataset.py              # PyTorch Dataset & DataLoader builder
│   ├── evaluation.py           # Metrics, Confusion Matrix, ROC-AUC, 5-Fold CV
│   ├── explainability.py       # Grad-CAM engine (layer4[-1])
│   ├── model.py                # ResNet-18 PyTorch module
│   ├── preprocessing.py        # ImagePreprocessor class
│   └── report.py               # PDF report engine (ReportLab / FPDF2)
├── app.py                      # Interactive Streamlit Web Application
├── train.py                    # Model training & fine-tuning script
├── requirements.txt            # Dependency requirements
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/arstriker/Parkinsons-Disease-Detection.git
cd Parkinsons-Disease-Detection

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset Index

```bash
python scripts/inspect_datasets.py
```

### 3. Run the Interactive Streamlit Web App

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test sample drawing images or upload custom spiral/wave images.

### 4. Re-Train & Evaluate Model

```bash
python train.py
```

---

## 🔬 Dataset Acknowledgments

- **Primary Dataset:** Kaggle "Parkinson's Drawings" dataset (spiral + wave drawings from healthy controls & PD patients).
- **Secondary Dataset:** NIATS-UFU HandPD / NewHandPD dataset (spiral and meander drawings captured via digitizing tablets).

---

## ⚠️ Medical Disclaimer

*This project is built strictly for research, educational, and point-of-care screening demonstration purposes. It is not an FDA-approved medical diagnostic tool and should not be used as a standalone diagnosis without qualified clinical supervision.*

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
