# 🐾 Cat Breed Identifier AI

### Enterprise-Grade Machine Learning Pipeline & Mobile Application

![Status](https://img.shields.io/badge/Status-Beta-orange)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![React Native](https://img.shields.io/badge/React_Native-Latest-blue)
![Expo](https://img.shields.io/badge/Expo-SDK_56-black)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20iOS-success)

> A production-ready Machine Learning Operations (MLOps) pipeline and a cross-platform React Native mobile application that trains, optimizes, and deploys a deep learning model capable of identifying 42 cat breeds directly on Android devices without requiring cloud connectivity.

---

# 📖 Table of Contents

* [Project Overview](#-project-overview)
* [Features](#-features)
* [System Architecture](#-system-architecture)
* [Technical Specifications](#-technical-specifications)
* [Training Strategy](#-training-strategy)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Execution Roadmap](#-execution-roadmap)
* [Technology Stack](#-technology-stack)
* [Model Performance](#-model-performance)
* [Future Improvements](#-future-improvements)
* [Intellectual Property Notice](#-intellectual-property-notice)
* [Author](#-author)

---

# 📌 Project Overview

Cat Breed Identifier AI is a complete end-to-end machine learning system designed to solve the fine-grained image classification challenge of distinguishing visually similar cat breeds, paired with a custom-built, full-screen camera application.

Unlike traditional AI applications that rely on remote API calls, this solution performs all inference locally on the user's device using a compressed TensorFlow Lite model and ultra-fast Nitro Modules, ensuring:

* ⚡ Optimized for real-time inference
* 🔒 Complete privacy (No images leave the device)
* 📶 Offline functionality (No internet required after installation)
* 📱 Mobile-first deployment optimized for mobile hardware

The project follows industry-standard MLOps practices including:

* Data validation
* Data preprocessing
* Transfer learning
* Hyperparameter tuning
* Model evaluation
* Quantization
* Edge deployment

---

# 🚀 Features

## 🧠 Deep Learning Classification

* Identifies **42 distinct cat breeds**
* Trained on **30,624 images**
* Two-phase transfer learning strategy
* TensorFlow Lite optimized deployment

---

## 📱 Live Native Camera Integration

* React Native + Expo Router application
* Custom full-screen camera interface
* Vision Camera v4 integration
* Nitro Modules architecture
* Direct frame inference pipeline
* Live prediction overlays
* Application branding and scanner UI

---

## ⚙️ Production Data Pipeline

* TensorFlow `tf.data` pipeline
* Automatic caching
* Automatic prefetching
* Dynamic class weighting
* Efficient batch processing

---

## 🔋 Edge AI Optimization

* TensorFlow Lite deployment
* Quantized model execution
* Reduced memory consumption
* Mobile CPU/NPU optimization
* Offline inference support

---

# 🏗️ System Architecture

```text
[ Cloud / Desktop MLOps Pipeline ]

Raw Images
    │
    ▼
Data Cleaning
    │
    ▼
Preprocessing
    │
    ▼
tf.data Pipeline
    │
    ▼
Transfer Learning (EfficientNetB0)
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

[ Mobile Edge Application ]

React Native App
    │
    ▼
Vision Camera v4
(YUV/RGB Frames)
    │
    ▼
Fast TFLite Execution
(Nitro Modules)
    │
    ▼
UI Overlay & Predictions
```

---

# 📊 Technical Specifications

| Component         | Details                    |
| ----------------- | -------------------------- |
| Architecture      | EfficientNetB0 (Base)      |
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
| Platform          | Android                    |

---

# 🔬 Training Strategy

## Phase 1 — Feature Extraction

The EfficientNetB0 backbone remains frozen while a custom classification head is trained.

### Configuration

* Frozen EfficientNetB0 layers
* Dense classification head
* Dropout (0.4)
* 15 epochs

### Objective

Preserve ImageNet-trained visual knowledge while adapting the model for cat breed classification.

---

## Phase 2 — Deep Fine-Tuning

The upper layers of EfficientNetB0 are unfrozen and retrained using a lower learning rate.

### Configuration

* Top layers unfrozen
* Learning Rate: 1e-5
* 40 epochs

### Objective

Learn subtle feline characteristics such as:

* Ear shape
* Fur texture
* Coat patterns
* Facial structure
* Eye characteristics

---

# ⚡ Model Optimization

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

## Dynamic Class Weighting

Handles dataset imbalance by assigning higher penalties to underrepresented breeds during training.

---

## Early Stopping

Automatically halts training when validation performance stops improving to reduce overfitting.

---

## Model Checkpointing

Continuously saves the best-performing model during training.

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
* Lower memory usage
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
├── mobile_app/
│   ├── assets/
│   ├── app/
│   ├── src/components/
│   ├── app.json
│   ├── eas.json
│   └── package.json
│
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

Install the generated APK from GitHub Releases.

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

## Tooling

* Git
* GitHub
* EAS Build
* VS Code

---

# 📈 Model Performance

> Final evaluation metrics will be published after completion of the training and validation phase.

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
* Automatic model updates
* Google Play Store release
* iOS support
* CI/CD automation improvements

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
