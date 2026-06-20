# 🐾 Cat Breed Identifier AI

### Enterprise-Grade Machine Learning Pipeline & Mobile Application

![Status](https://img.shields.io/badge/Status-Beta-orange)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![React Native](https://img.shields.io/badge/React_Native-Latest-blue)
![Expo](https://img.shields.io/badge/Expo-SDK_56-black)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20iOS-success)

> A production-ready Machine Learning Operations (MLOps) pipeline and a cross-platform React Native mobile application that trains, optimizes, and deploys a deep learning model capable of identifying 42 cat breeds directly on Android and iOS devices without requiring cloud connectivity.

---

# 📖 Table of Contents

* [Project Overview](#-project-overview)
* [Features](#-features)
* [System Architecture](#-system-architecture)
* [Technical Specifications](#-technical-specifications)
* [Training Strategy](#-training-strategy)
* [Model & Pipeline Optimization](#-model--pipeline-optimization)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Execution Roadmap](#-execution-roadmap)
* [Technology Stack](#-technology-stack)
* [Model Performance](#-model-performance)
* [Future Improvements](#-future-improvements)
* [Intellectual Property Notice](#-intellectual-property-notice)

---

# 📌 Project Overview

Cat Breed Identifier AI is a complete end-to-end machine learning ecosystem designed to solve the fine-grained image classification challenge of distinguishing visually similar cat breeds.

The project combines:

* A complete TensorFlow-based MLOps training pipeline
* Transfer learning with EfficientNetB0
* TensorFlow Lite optimization
* A React Native mobile application
* Real-time camera inference using Vision Camera and Nitro Modules

Unlike traditional AI applications that rely on cloud APIs, all inference is performed directly on the device.

### Key Benefits

* ⚡ Optimized for real-time inference
* 🔒 Complete privacy
* 📶 Offline functionality
* 📱 Mobile-first deployment
* 🧠 On-device AI processing

---

# 🚀 Features

## 🧠 Deep Learning Classification

* Identifies 42 distinct cat breeds
* Trained on 30,624 images
* Two-phase transfer learning strategy
* TensorFlow Lite optimized deployment
* Offline breed prediction

---

## 📱 Live Native Camera Integration

* React Native + Expo Router
* Vision Camera v4 integration
* Nitro Modules architecture
* Direct frame inference pipeline
* Real-time prediction overlays

---

## ⚙️ Production Data Pipeline

* TensorFlow tf.data API
* Automatic caching
* Automatic prefetching
* Dynamic class weighting
* GPU-optimized batch processing

---

## 🔋 Edge AI Optimization

* TensorFlow Lite conversion
* Integer quantization
* Reduced memory consumption
* Faster mobile inference
* Lower battery usage

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     Raw Dataset     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Cleaning     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Preprocessing     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  tf.data Pipeline   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   EfficientNetB0    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Fine-Tuning      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ TensorFlow Lite     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ React Native App    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Vision Camera v4    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Nitro Modules       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Live Predictions    │
                    └─────────────────────┘
```

---

# 📊 Technical Specifications

| Component         | Details                    |
| ----------------- | -------------------------- |
| Architecture      | EfficientNetB0             |
| Classes           | 42 Cat Breeds              |
| Dataset Size      | 30,624 Images              |
| Training Images   | 24,500                     |
| Validation Images | 6,124                      |
| Input Resolution  | 224 × 224 RGB              |
| ML Framework      | TensorFlow / Keras         |
| Mobile Framework  | React Native + Expo SDK 56 |
| Camera Framework  | Vision Camera v4           |
| Native Runtime    | Nitro Modules              |
| Deployment Format | TensorFlow Lite (.tflite)  |
| Target Platforms  | Android & iOS              |

---

# 🔬 Training Strategy

## Phase 1 — Feature Extraction

The EfficientNetB0 backbone remains frozen while a custom classification head is trained.

### Configuration

* Frozen EfficientNetB0 layers
* Dense classification head
* Dropout: 0.4
* Epochs: 15

### Goal

Retain ImageNet knowledge while adapting the model for cat breed recognition.

---

## Phase 2 — Deep Fine-Tuning

The upper 50 layers are unfrozen and trained using a lower learning rate.

### Configuration

* Top 50 layers unfrozen
* Learning Rate: 1e-5
* Epochs: 40

### Goal

Learn subtle feline characteristics including:

* Ear shape
* Fur texture
* Coat patterns
* Facial structure
* Eye characteristics

---

# ⚡ Model & Pipeline Optimization

## TensorFlow Data Pipeline

```python
dataset.cache()
dataset.prefetch(tf.data.AUTOTUNE)
```

Benefits:

* Faster training throughput
* Reduced CPU bottlenecks
* Better GPU utilization

---

## Dataset Imbalance Handling

* Dynamic class weighting
* Minority breed protection
* Reduced prediction bias

---

## Early Stopping

```python
EarlyStopping(
    monitor="val_accuracy",
    patience=6
)
```

Prevents overfitting and unnecessary training cycles.

---

## Model Checkpointing

Automatically saves the best-performing model during training.

---

## TensorFlow Lite Quantization

```python
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]
```

Benefits:

* Smaller model size
* Faster inference
* Lower memory consumption
* Reduced battery usage

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
├── mobile_app/
│   ├── assets/models/
│   ├── app/
│   ├── src/components/
│   ├── app.json
│   ├── eas.json
│   └── package.json
│
├── requirements.txt
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Cat-Breed-Identifier-AI.git
cd Cat-Breed-Identifier-AI
```

---

## Python Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Mobile Application

```bash
cd mobile_app
npm install
```

Start development server:

```bash
npx expo start -c
```

---

# 🚀 Execution Roadmap

## Phase 1 — Dataset Preparation

```bash
python src/clean_data.py
python src/preprocess.py
```

---

## Phase 2 — Model Training

```bash
python src/train.py
```

---

## Phase 3 — Evaluation

```bash
python src/plot_history.py
python src/predict.py
```

---

## Phase 4 — TensorFlow Lite Conversion

```bash
python src/convert_tflite.py
python src/verify_tflite.py
```

---

## Phase 5 — Mobile Deployment

```bash
cd mobile_app
eas build --profile preview --platform android
```

Install via EAS QR Code or GitHub Releases.

---

# 🛠️ Technology Stack

## Machine Learning

* Python
* TensorFlow
* Keras
* TensorFlow Lite
* NumPy
* Matplotlib

## Mobile Development

* React Native
* Expo SDK 56
* Expo Router
* TypeScript
* Vision Camera v4
* Fast TFLite
* Nitro Modules

## DevOps & Tooling

* Git
* GitHub
* EAS Build
* VS Code

---

# 📈 Model Performance

> Final metrics will be published after completion of the validation and optimization phase.

| Metric                 | Value |
| ---------------------- | ----- |
| Validation Accuracy    | TBD   |
| Precision              | TBD   |
| Recall                 | TBD   |
| F1 Score               | TBD   |
| Keras Model Size       | TBD   |
| TFLite Model Size      | TBD   |
| Average Inference Time | TBD   |

---

# 🔮 Future Improvements

* Support additional cat breeds
* Multi-cat detection
* Breed confidence visualization
* Breed information cards
* Automated model updates
* iOS App Store release
* Google Play Store release
* CI/CD automation

---

# 🔒 Intellectual Property Notice

## PROPRIETARY AND CONFIDENTIAL

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
Specialization in Artificial Intelligence

---

### ⭐ If you found this project interesting, please consider starring the repository.
