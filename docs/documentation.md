# Technical Documentation
## Automated Chest X-Ray Disease Classifier

---

## 1. Project Overview

This project implements an end-to-end medical image classification pipeline
that automatically detects 14 chest diseases from X-ray images. It combines
data engineering (Apache Airflow, DVC) with deep learning (EfficientNet-B0,
PyTorch) to build a production-ready system.

---

## 2. Data Engineering Pipeline

### 2.1 Airflow DAG
File: `airflow_dags/chest_xray_dag.py`

The DAG runs on a daily schedule and executes 5 tasks in sequence:

**Task 1: validate_data**
- Checks labels CSV exists and has required columns
- Validates all disease labels against known 14-class list
- Checks for null values and data integrity issues
- Raises ValueError if validation fails

**Task 2: preprocess_data**
- Creates multi-hot encoding for all 14 disease labels
- Normalizes patient age using z-score normalization
- Encodes patient gender as binary (M=0, F=1)
- Saves processed CSV to data/processed/

**Task 3: augment_data**
- Saves augmentation configuration as JSON
- Settings: horizontal flip, 10° rotation, brightness ±20%,
  zoom 10%, target size 224×224
- Normalization: ImageNet mean/std ([0.485, 0.456, 0.406])

**Task 4: split_data**
- Shuffles dataset with random seed 42 (reproducible)
- Splits: 70% train / 15% validation / 15% test
- Saves separate CSV files for each split

**Task 5: version_data**
- Tracks all data splits with DVC
- Enables dataset versioning and reproducibility
- Links data versions to git commits

### 2.2 DVC Versioning
DVC tracks the data directory separately from git.
Data changes are tracked in `.dvc` files which ARE committed to git.
Actual data files are in `.gitignore`.

---

## 3. Model Architecture

### 3.1 EfficientNet-B0 Backbone
- Pretrained on ImageNet (1.2M images, 1000 classes)
- Input: 224×224×3 RGB images
- Feature extraction: 1280-dimensional feature vector

### 3.2 Custom Classification Head
```
Dropout(0.3)
Linear(1280 → 512)
ReLU()
Dropout(0.2)
Linear(512 → 14)
```

### 3.3 Multi-Label Classification
- Output: 14 sigmoid probabilities (one per disease)
- Loss: BCEWithLogitsLoss (binary cross-entropy per label)
- Threshold: 0.5 for positive prediction

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Batch size | 32 |
| Epochs | 10 |
| LR scheduler | StepLR (step=5, gamma=0.1) |
| GPU | NVIDIA T4 (Google Colab) |

---

## 5. Evaluation Metrics

### 5.1 AUC-ROC
- Computed per disease class
- Macro-averaged across all 14 classes
- Mean AUC: 0.8474 (target: >0.85 on full dataset)

### 5.2 Grad-CAM Visualization
Gradient-weighted Class Activation Mapping shows which
regions of the X-ray the model focuses on for each disease.
- Target layer: EfficientNet last convolutional block
- Output: heatmap overlaid on original X-ray

---

## 6. Dataset Details

### NIH ChestX-ray14
- 112,120 frontal-view X-rays
- 30,805 unique patients
- 14 disease labels (multi-label)
- Image size: varies (resized to 224×224)
- Source: NIH Clinical Center (public domain)

### Disease Classes
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

---

## 7. File Reference

| File | Description |
|------|-------------|
| `src/download_sample.py` | Creates sample dataset structure |
| `src/preprocess.py` | All 5 pipeline task functions |
| `airflow_dags/chest_xray_dag.py` | Airflow DAG definition |
| `notebooks/chest_xray_training.ipynb` | Full training notebook |
| `data/raw/labels.csv` | Raw NIH format labels |
| `data/processed/labels_processed.csv` | Multi-hot encoded labels |
| `data/processed/augmentation_config.json` | Augmentation settings |

---

## 8. How to Reproduce Results

1. Run `python src/download_sample.py` to set up data structure
2. Run `python src/preprocess.py` to run full pipeline locally
3. Open `notebooks/chest_xray_training.ipynb` in Google Colab
4. Enable T4 GPU runtime
5. Run all cells in order
6. Results saved to `/content/` on Colab