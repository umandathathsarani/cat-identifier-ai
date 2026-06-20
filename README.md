# 🐾 Cat Breed Identifier AI

### Enterprise-Grade Machine Learning Pipeline & Cross-Platform Mobile Application

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![React Native](https://img.shields.io/badge/React_Native-Latest-blue)
![Expo](https://img.shields.io/badge/Expo-SDK_56-black)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-Mobile-green)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20iOS-success)

> A production-ready Machine Learning Operations (MLOps) pipeline and mobile AI application that trains, optimizes, and deploys a deep learning model capable of identifying 42 cat breeds directly on Android and iOS devices without requiring cloud connectivity.

---

## 📖 Table of Contents

* [Project Overview](#-project-overview)
* [Features](#-features)
* [System Architecture](#-system-architecture)
* [Technical Specifications](#-technical-specifications)
* [Training Strategy](#-training-strategy)
* [Model Optimization](#-model--pipeline-optimization)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Execution Roadmap](#-execution-roadmap)
* [Technology Stack](#-technology-stack)
* [Results](#-results)
* [Future Improvements](#-future-improvements)
* [Intellectual Property Notice](#-intellectual-property-notice)

---

# 📌 Project Overview

Cat Breed Identifier AI is an end-to-end machine learning ecosystem designed to solve the fine-grained image classification problem of distinguishing visually similar cat breeds.

The project combines:

* A complete TensorFlow-based MLOps training pipeline
* Transfer learning with EfficientNetB0
* TensorFlow Lite model optimization
* A React Native mobile application
* Real-time camera inference using Vision Camera and JSI-based processing

Unlike cloud-dependent AI systems, all inference is performed locally on the device, providing:

* ⚡ Near-zero latency predictions
* 🔒 Complete user privacy
* 📶 Offline functionality
* 📱 Mobile-first deployment

---

# 🚀 Features

## 🧠 Deep Learning Classification

* Identifies 42 cat breeds
* Trained on 30,624 images
* Two-phase transfer learning strategy
* Optimized for edge deployment

## 📱 Mobile AI Scanner

* React Native + Expo architecture
* Vision Camera v4 integration
* Fast TensorFlow Lite inference
* Real-time breed predictions
* Native Android and iOS support

## ⚙️ Production Data Pipeline

* TensorFlow tf.data API
* Dynamic class weighting
* Automatic caching
* Automatic prefetching
* GPU-optimized data loading

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
                     │     Raw Images      │
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
                     │ Live Camera Scanner │
                     └─────────────────────┘
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
| Input Resolution  | 224 × 224 RGB             |
| ML Framework      | TensorFlow / Keras        |
| Mobile Framework  | React Native + Expo       |
| Deployment Format | TensorFlow Lite (.tflite) |
| Target Platforms  | Android & iOS             |

---

# 🔬 Training Strategy

## Phase 1 – Feature Extraction

The EfficientNetB0 backbone remains frozen while a custom classification head is trained.

### Configuration

* Frozen EfficientNetB0 layers
* Dense classification head
* Dropout: 0.4
* Epochs: 15

### Goal

Retain ImageNet knowledge while adapting the model for cat breed recognition.

---

## Phase 2 – Deep Fine-Tuning

The upper 50 layers are unfrozen and trained using a lower learning rate.

### Configuration

* Top 50 layers unfrozen
* Learning rate: 1e-5
* Epochs: 40

### Goal

Learn subtle feline features including:

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

* Faster training
* Reduced CPU bottlenecks
* Improved GPU utilization

## Dataset Imbalance Handling

* Dynamic class weighting
* Minority breed protection
* Reduced prediction bias

## Early Stopping

```python
EarlyStopping(
    monitor="val_accuracy",
    patience=6
)
```

## Model Checkpointing

Automatically saves the best-performing model throughout training.

## TensorFlow Lite Quantization

```python
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]
```

Benefits:

* Smaller model size
* Faster inference
* Lower battery usage

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
│   ├── src/
│   ├── app.json
│   ├── eas.json
│   └── package.json
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/cat-identifier-ai.git
cd cat-identifier-ai
```

## Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Mobile Application

```bash
cd mobile_app
npm install
```

---

# 🚀 Execution Roadmap

## Dataset Preparation

```bash
python src/clean_data.py
python src/preprocess.py
```

## Model Training

```bash
python src/train.py
```

## Evaluation

```bash
python src/plot_history.py
python src/predict.py
```

## TensorFlow Lite Conversion

```bash
python src/convert_tflite.py
python src/verify_tflite.py
```

## Mobile Build

```bash
cd mobile_app
eas build --profile preview --platform android
```

---

# 📈 Results

| Metric              | Value |
| ------------------- | ----- |
| Validation Accuracy | TBD   |
| Precision           | TBD   |
| Recall              | TBD   |
| F1 Score            | TBD   |
| Model Size (Keras)  | TBD   |
| Model Size (TFLite) | TBD   |

> Update this section once final training is completed.

---

# 🔮 Future Improvements

* Support additional cat breeds
* Multi-pet classification
* Breed confidence visualization
* On-device model updates
* iOS App Store deployment
* Android Play Store deployment
* Automated CI/CD pipelines

---

# 🛠️ Technology Stack

### Machine Learning

* Python
* TensorFlow
* Keras
* TensorFlow Lite
* NumPy
* Matplotlib

### Mobile Development

* React Native
* Expo
* TypeScript
* Vision Camera
* Fast-TFLite
* Reanimated Worklets

### DevOps & Tooling

* Git
* GitHub
* EAS Build
* VS Code

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
Specialization: Artificial Intelligence

---

### ⭐ If you found this project interesting, consider starring the repository.
