# 🏞️ Large-Scale Scene Classification with EfficientNet-B2

> A deep learning image classification pipeline for recognizing **397 scene categories** across **60,000+ images**, built with PyTorch and transfer learning.

---

## 📋 Overview

Scene recognition is a fundamental problem in computer vision with wide-ranging applications — from autonomous navigation and robotics to photo organization and content-based image retrieval. Unlike object classification, scene classification requires understanding the **holistic composition** of an image: spatial layout, texture, contextual objects, and their relationships.

This project tackles **fine-grained scene classification** at scale using the **SUN 397** benchmark, one of the most comprehensive scene understanding datasets available. With 397 categories spanning indoor, outdoor, natural, and man-made environments, the task presents a significant challenge in high-cardinality visual recognition.

The pipeline leverages **EfficientNet-B2** with transfer learning from ImageNet, combined with modern training techniques including mixed-precision training, label smoothing, and OneCycleLR scheduling, to efficiently learn discriminative scene representations.

---

## ✨ Key Features

- **End-to-End Pipeline** — From raw image preprocessing to final inference, fully implemented in a single reproducible notebook
- **High-Cardinality Classification** — 397 distinct scene categories with stratified train/validation splitting
- **Transfer Learning** — EfficientNet-B2 pretrained on ImageNet, fine-tuned for scene recognition
- **Mixed-Precision Training** — CUDA AMP for faster training with reduced memory footprint
- **Data Augmentation** — Random horizontal flips and color jitter for improved generalization
- **Test-Time Augmentation (TTA)** — Horizontal flip TTA during inference for more robust predictions
- **OneCycleLR Scheduling** — Cosine-annealing learning rate policy with warm-up for stable convergence
- **Label Smoothing** — Cross-entropy loss with label smoothing (ε=0.1) to reduce overconfidence
- **Comprehensive Evaluation** — Accuracy, Macro F1, and Weighted F1 metrics tracked per epoch

---

## 🛠️ Tech Stack

| Component       | Technology                          |
| --------------- | ----------------------------------- |
| Language        | Python 3.10+                        |
| Deep Learning   | PyTorch                             |
| Model Library   | [timm](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models) |
| Architecture    | EfficientNet-B2                     |
| Data Processing | Pandas, NumPy, Pillow               |
| ML Utilities    | scikit-learn (splitting, metrics)   |
| Training        | CUDA AMP, AdamW, OneCycleLR        |
| Environment     | Kaggle / Google Colab (GPU)         |

---

## 🔄 Project Pipeline

```
Raw Images (60K+)
    │
    ▼
┌──────────────────────┐
│  1. Image Resizing   │  Resize all images to 224×224 (JPEG, quality=90)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  2. Data Loading     │  Load TRAIN.csv, build label mapping (397 classes)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  3. Stratified Split │  90% train / 10% validation (preserving class distribution)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  4. Augmentation     │  RandomHorizontalFlip + ColorJitter (train only)
│     & Normalization  │  ImageNet normalization for all splits
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  5. Model Setup      │  EfficientNet-B2 (pretrained) + new 397-class head
│                      │  AdamW optimizer + OneCycleLR + Label Smoothing
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  6. Training Loop    │  30 epochs with mixed-precision (AMP)
│                      │  Best model checkpoint saved on validation accuracy
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  7. Inference + TTA  │  Load best checkpoint, predict with horizontal flip TTA
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  8. Export Results    │  Save predictions to CSV
└──────────────────────┘
```

### Step-by-Step Details

1. **Image Resizing** — All 60K+ raw images are pre-resized to 224×224 pixels and saved as JPEG (quality=90). This one-time preprocessing step significantly reduces I/O overhead during training.

2. **Data Loading** — Training labels are loaded from `TRAIN.csv`. A deterministic label-to-index mapping is constructed from the sorted unique labels, ensuring reproducibility.

3. **Stratified Train/Validation Split** — The training set is split 90/10 using `train_test_split` with `stratify` to maintain the original class distribution in both partitions.

4. **Augmentation & Normalization** — Training images receive random horizontal flips and color jitter (brightness, contrast, saturation, hue). All images are normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

5. **Model Setup** — An EfficientNet-B2 backbone is loaded with ImageNet-pretrained weights via `timm`. The classifier head is replaced with a linear layer mapping to 397 output classes. The optimizer is AdamW with weight decay regularization.

6. **Training** — The model trains for 30 epochs using mixed-precision (CUDA AMP) with GradScaler. The learning rate follows a OneCycleLR schedule with 10% warm-up. The best model (by validation accuracy) is checkpointed.

7. **Inference with TTA** — At test time, predictions are generated for both the original and horizontally-flipped versions of each image. The logits are averaged before taking the argmax, providing more robust predictions.

8. **Export** — Final predictions are mapped back to original label names and saved as a CSV file.

---

## 🧠 Model Details

### Why EfficientNet-B2?

**EfficientNet** (Tan & Le, 2019) introduces a principled approach to scaling convolutional neural networks by uniformly scaling depth, width, and resolution using a compound coefficient. The B2 variant strikes an optimal balance:

| Property           | Value    |
| ------------------ | -------- |
| Parameters         | ~8.3M    |
| Input Resolution   | 224×224  |
| Top-1 Acc (ImageNet) | ~80.1% |

- **Efficiency** — Fewer parameters than ResNet-50 with better accuracy, enabling faster training on limited GPU budgets
- **Transfer Learning Fit** — ImageNet pretraining provides strong general-purpose visual features that transfer well to scene classification
- **Scalability** — The EfficientNet family provides a clear upgrade path (B3→B7) if more compute becomes available

### Transfer Learning Strategy

The full EfficientNet-B2 backbone is loaded with pretrained ImageNet weights. Only the final classifier head is replaced with a new `nn.Linear(in_features, 397)` layer. All parameters are fine-tuned end-to-end (no layer freezing), allowing the backbone features to adapt to scene-specific patterns while leveraging the strong initialization from ImageNet.

---

## 📊 Results

| Metric              | Score    |
| -------------------- | -------- |
| **Validation Accuracy** | **68.30%** |
| **F1 Score (Macro)**    | **0.6176** |
| **F1 Score (Weighted)** | **0.6761** |
| Training Accuracy    | 99.99%   |
| Best Epoch           | 22 / 30  |

> **Note:** Results are on a 10% stratified holdout from the training set (3,729 images). Given 397 fine-grained scene categories, a ~68% accuracy represents strong performance — random chance would yield ~0.25%.

### Training Progression Highlights

| Epoch | Train Acc | Val Acc | F1 Macro | F1 Weighted |
|-------|-----------|---------|----------|-------------|
| 1     | 11.38%    | 30.57%  | 0.1071   | 0.2133      |
| 5     | 87.90%    | 63.88%  | 0.5609   | 0.6296      |
| 10    | 98.80%    | 65.73%  | 0.5834   | 0.6489      |
| 15    | 99.68%    | 66.69%  | 0.6003   | 0.6606      |
| 20    | 99.95%    | 67.58%  | 0.6137   | 0.6696      |
| **22**| **99.99%**| **68.30%** | **0.6176** | **0.6761** |

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/scene-classification-efficientnet.git
cd scene-classification-efficientnet

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset

This project uses the **SUN 397** scene classification dataset (~22 GB). Place the dataset in the `data/` directory:

```
data/
├── images/          # 62K+ scene images (JPG/PNG)
├── TRAIN.csv        # IMAGE → LABEL mapping (37,287 samples)
└── TEST.csv         # IMAGE filenames for inference (24,858 samples)
```

> **Note:** The dataset is not included in this repository due to its size. See the [SUN 397 dataset page](https://vision.princeton.edu/projects/2010/SUN/) for download instructions.

### Training

Open `notebooks/scene_classification.ipynb` in Jupyter or run on [Kaggle](https://www.kaggle.com/) / [Google Colab](https://colab.research.google.com/) with GPU acceleration.

Update the file paths in the **Configuration** cell to point to your local data directory, then run all cells sequentially.

---

## 📁 Project Structure

```
scene-classification-efficientnet/
│
├── data/                          # Dataset directory (not tracked by git)
│   ├── images/                    # Raw scene images
│   ├── TRAIN.csv                  # Training labels
│   └── TEST.csv                   # Test image filenames
│
├── notebooks/
│   └── scene_classification.ipynb # Main training & evaluation notebook
│
├── models/                        # Saved model checkpoints
│   └── README.md                  # Placeholder with instructions
│
├── outputs/                       # Predictions and results
│   └── README.md                  # Placeholder with instructions
│
├── src/                           # Modular Python scripts (future use)
│   └── README.md                  # Placeholder with instructions
│
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔮 Future Improvements

- [ ] **Model Comparison** — Benchmark against ResNet-50, ConvNeXt, and Vision Transformers (ViT, DeiT)
- [ ] **Ensemble Methods** — Combine predictions from multiple architectures for improved accuracy
- [ ] **Hyperparameter Tuning** — Systematic search over learning rates, augmentation strategies, and schedulers using Optuna
- [ ] **Advanced Augmentation** — Integrate CutMix, MixUp, and RandAugment for stronger regularization
- [ ] **Deployment** — Package the trained model as a REST API using FastAPI or deploy via ONNX Runtime
- [ ] **Grad-CAM Visualization** — Add interpretability analysis to understand which image regions drive classification decisions
- [ ] **Class Imbalance Handling** — Investigate focal loss or class-weighted sampling for underrepresented categories
- [ ] **Progressive Resizing** — Start training at lower resolution and progressively increase for faster convergence

---

## 🏆 Hackathon Project

This project was developed as part of an ML-focused hackathon by a collaborative team. The objective was to build a scalable scene classification system capable of recognizing 397 scene categories from over 60,000 images using modern deep learning and transfer learning techniques.

The project leverages EfficientNet-B2, mixed-precision training (AMP), label smoothing, OneCycleLR scheduling, and Test Time Augmentation (TTA) to improve performance on the challenging SUN397 benchmark dataset.

---

## 👥 Authors & Contributors

### Harshita Pokhariya

* Model experimentation and evaluation
* Dataset preprocessing and preparation
* Performance analysis using Accuracy and F1 metrics
* Documentation and project presentation
* Training pipeline optimization and testing

GitHub: https://github.com/harshitapokhariya

### Samar Jamal

* Project architecture and implementation
* Deep learning pipeline development
* Model training and optimization
* Integration of EfficientNet-B2 and training strategies

GitHub: https://github.com/Samarjamal326

---

## 🤝 Collaboration Note

This repository is maintained as part of a team-developed hackathon project. Contributions were made collaboratively across data preparation, model development, experimentation, evaluation, optimization, and documentation. The project demonstrates practical experience in large-scale image classification, transfer learning, and deep learning model evaluation.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---
