# Contributing to Cat Breed Identifier AI

Thank you for your interest in the Cat Breed Identifier AI project.

This repository contains a production-oriented Machine Learning Operations (MLOps) pipeline and a cross-platform mobile application designed for on-device cat breed classification.

Before participating in development activities, please read this document carefully.

---

# 📋 Project Structure

This repository is organized as a monorepo containing two primary environments.

## 1. Machine Learning Pipeline

Location:

```text
/
```

Technology Stack:

* Python
* TensorFlow
* Keras
* TensorFlow Lite
* NumPy
* Matplotlib

Responsibilities:

* Data validation
* Data preprocessing
* Model training
* Transfer learning
* Model evaluation
* TensorFlow Lite conversion

---

## 2. Mobile Edge Application

Location:

```text
/mobile_app
```

Technology Stack:

* React Native
* Expo
* TypeScript
* Vision Camera
* Fast TFLite
* Native C++ JSI Modules

Responsibilities:

* Camera integration
* Real-time image processing
* TensorFlow Lite inference
* Mobile UI/UX
* Android and iOS deployment

---

# 💻 Development Environment Setup

## Machine Learning Pipeline

### Prerequisites

* Python 3.10+
* Git

### Clone Repository

```bash
git clone https://github.com/umandathathsarani/cat-identifier-ai.git
cd cat-identifier-ai
```

### Create Virtual Environment

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Mobile Application

### Prerequisites

* Node.js 20+
* Android Studio
* Xcode (macOS only)
* Expo CLI

### Install Dependencies

```bash
cd mobile_app
npm install
```

### Start Development Server

```bash
npx expo start -c
```

---

# 🌿 Branching Strategy

All development work should be performed in dedicated feature branches.

Branch naming convention:

```text
feat/feature-name
fix/bug-description
docs/documentation-update
chore/task-name
```

Examples:

```text
feat/ios-tflite-support
feat/breed-confidence-overlay
fix/android-camera-crash
docs/readme-update
chore/dependency-upgrade
```

---

# 📝 Commit Message Convention

This repository follows the Conventional Commits specification.

## Feature

```text
feat: add iOS support for TensorFlow Lite delegate
```

## Bug Fix

```text
fix: resolve camera permission issue on Android 14
```

## Documentation

```text
docs: update installation instructions
```

## Maintenance

```text
chore: upgrade Expo SDK dependencies
```

---

# 🧪 Testing Requirements

Before submitting code for review, verify the following:

## Machine Learning Pipeline

```bash
python src/clean_data.py
python src/preprocess.py
python src/train.py
```

Ensure:

* No runtime exceptions
* Successful model export
* Successful TensorFlow Lite conversion

---

## Mobile Application

Verify:

* Application launches successfully
* Camera permissions function correctly
* TensorFlow Lite model loads successfully
* Real-time predictions execute without crashes

---

# 🚀 Code Review Process

Development contributions are reviewed according to:

* Code quality
* Performance impact
* Architecture consistency
* Documentation completeness
* Mobile compatibility
* Model accuracy impact

All submissions must pass review before integration into the main branch.

---

# 🔒 Intellectual Property Notice

This repository and its contents are proprietary intellectual property.

Submission of code, suggestions, pull requests, or architectural recommendations does not transfer ownership of the project or any associated intellectual property.

All accepted contributions become part of the repository and are governed by the repository's licensing and copyright terms.

---

# 📞 Contact

For development inquiries, collaboration requests, or research discussions, please contact the repository owner through GitHub.

Thank you for your interest in Cat Breed Identifier AI.
