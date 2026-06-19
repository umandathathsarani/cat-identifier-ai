# 🐾 Cat Breed Identifier AI

### Enterprise-Grade Machine Learning Pipeline for Mobile Edge Deployment

> A production-ready Machine Learning Operations (MLOps) pipeline that trains, optimizes, and deploys a deep learning model capable of identifying 42 cat breeds directly on Android and iOS devices without requiring cloud connectivity.

---

## 📌 Project Overview

Cat Breed Identifier AI is a complete end-to-end machine learning system designed to solve the fine-grained image classification challenge of distinguishing visually similar cat breeds.

Unlike traditional AI applications that rely on remote API calls, this solution performs all inference locally on the user's device using a compressed TensorFlow Lite model, ensuring:

* ⚡ Near-zero latency predictions
* 🔒 Complete privacy
* 📶 Offline functionality
* 📱 Mobile-first deployment

The project follows industry-standard MLOps practices including data validation, preprocessing, transfer learning, hyperparameter tuning, model evaluation, quantization, and edge deployment.

---

# 🚀 Features

### 🧠 Deep Learning Classification

* Identifies **42 distinct cat breeds**
* Trained on **30,624 images**
* Uses transfer learning for improved accuracy
* Optimized for mobile deployment

### ⚙️ Production Data Pipeline

* TensorFlow `tf.data` pipeline
* Automatic caching and prefetching
* Dynamic class weighting for dataset imbalance handling
* Efficient batch processing

### 📈 Model Optimization

* Two-phase transfer learning strategy
* Early stopping
* Model checkpointing
* Fine-tuning of upper EfficientNet layers
* Quantization for reduced model size

### 📱 Mobile Edge Deployment

* TensorFlow Lite conversion
* Integer quantization
* Android and iOS compatible
* Offline inference support

---

# 🏗️ System Architecture

```text
Raw Images
     │
     ▼
Data Cleaning
     │
     ▼
Data Preprocessing
     │
     ▼
tf.data Pipeline
(Cache + Prefetch)
     │
     ▼
EfficientNetB0
Transfer Learning
     │
     ▼
Fine-Tuning
     │
     ▼
Model Evaluation
     │
     ▼
TensorFlow Lite Conversion
     │
     ▼
Mobile Deployment
```

---

# 📊 Technical Specifications

| Component         | Details                   |
| ----------------- | ------------------------- |
| Architecture      | EfficientNetB0            |
| Baseline Model    | MobileNetV3-Small         |
| Classes           | 42 Cat Breeds             |
| Dataset Size      | 30,624 Images             |
| Training Images   | 24,500                    |
| Validation Images | 6,124                     |
| Input Size        | 224 × 224 RGB             |
| Framework         | TensorFlow / Keras        |
| Deployment Format | TensorFlow Lite (.tflite) |
| Platform          | Android & iOS             |

---

# 🔬 Training Strategy

## Phase 1 – Feature Extraction

The EfficientNetB0 backbone remains frozen while a custom classification head is trained.

### Configuration

* Frozen EfficientNetB0 layers
* Dense classification head
* Dropout: 0.4
* Epochs: 15
* Transfer learning initialization

### Goal

Preserve ImageNet-learned visual features while adapting the model to cat breed classification.

---

## Phase 2 – Deep Fine-Tuning

The upper layers of EfficientNetB0 are unfrozen and retrained with a significantly reduced learning rate.

### Configuration

* Top 50 layers unfrozen
* Learning rate: 1e-5
* Epochs: 40
* Fine-grained breed optimization

### Goal

Allow the model to learn subtle feline characteristics such as:

* Ear shape
* Fur texture
* Coat patterns
* Facial structure
* Eye characteristics

---

# ⚡ Data Pipeline Optimization

The project uses TensorFlow's optimized input pipeline:

```python
dataset.cache()
dataset.prefetch(tf.data.AUTOTUNE)
```

Benefits:

* Faster training throughput
* Reduced CPU bottlenecks
* Better GPU utilization
* Lower disk I/O overhead

---

# ⚖️ Dataset Imbalance Handling

To address uneven breed distributions, dynamic class weighting is applied.

### Strategy

* Majority classes capped at 1,000 images
* Minority classes guaranteed a minimum of 150 images
* Weighted loss calculation during training

Benefits:

* Reduced prediction bias
* Better minority breed recognition
* Improved overall model fairness

---

# 🤖 Training Callbacks

## Early Stopping

```python
monitor="val_accuracy"
patience=6
```

Automatically stops training when validation performance stops improving.

---

## Model Checkpointing

Continuously saves the best-performing model during training.

Benefits:

* Prevents loss of optimal weights
* Enables recovery from interruptions
* Preserves best validation accuracy

---

# 📱 TensorFlow Lite Optimization

The trained Keras model is compressed using TensorFlow Lite quantization.

```python
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]
```

Benefits:

* Smaller model size
* Faster inference
* Lower memory consumption
* Better battery efficiency

---

# 📂 Project Structure

```text
Cat-Breed-Identifier-AI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── efficientnet_cat_identifier.keras
│   ├── model.tflite
│   ├── labels.txt
│   └── training_history.json
│
├── plots/
│   └── training_performance.png
│
├── src/
│   ├── clean_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── plot_history.py
│   ├── convert_tflite.py
│   └── verify_tflite.py
│
├── README.md
└── requirements.txt
```

---

# 🚀 Execution Roadmap

## Phase 1 — Dataset Preparation

```bash
python src/clean_data.py
python src/preprocess.py
```

### Purpose

* Remove corrupted images
* Validate dataset integrity
* Create optimized TensorFlow datasets

---

## Phase 2 — Model Training

```bash
python src/train.py
```

### Purpose

* Feature extraction training
* Deep fine-tuning
* Weight optimization

---

## Phase 3 — Evaluation

```bash
python src/plot_history.py
python src/predict.py
```

### Purpose

* Visualize learning curves
* Verify model predictions

---

## Phase 4 — Mobile Deployment

```bash
python src/convert_tflite.py
python src/verify_tflite.py
```

### Purpose

* Generate TensorFlow Lite model
* Validate edge-device inference

---

# 📈 Outputs

Generated artifacts include:

| Artifact                   | Description            |
| -------------------------- | ---------------------- |
| `.keras` Model             | Full desktop model     |
| `.tflite` Model            | Mobile-ready model     |
| `labels.txt`               | Class labels           |
| `training_history.json`    | Training metrics       |
| `training_performance.png` | Accuracy & loss curves |

---

# 🛠️ Technology Stack

* Python
* TensorFlow
* Keras
* TensorFlow Lite
* NumPy
* Matplotlib
* JSON
* tf.data API

---

# 🔒 Intellectual Property Notice

**PROPRIETARY AND CONFIDENTIAL**

Copyright © 2026 Umanda Thathsarani. All Rights Reserved.

This repository, its architecture, source code, training methodologies, datasets, documentation, and trained model weights are proprietary intellectual property.

### Restrictions

* ❌ No copying
* ❌ No redistribution
* ❌ No reproduction
* ❌ No derivative works
* ❌ No commercial use without written permission

### Academic Integrity

This project was developed as original work for academic and educational purposes.

Any unauthorized reuse, duplication, or submission of this work in academic environments constitutes plagiarism and may result in disciplinary and legal action.

---

# 👨‍💻 Author

**Umanda Thathsarani**

Bachelor of Science (Honours) in Information Technology
Specialization: Artificial Intelligence

---

### ⭐ If you found this project interesting, please consider starring the repository.
