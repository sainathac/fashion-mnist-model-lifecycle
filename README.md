# 👗 Fashion-MNIST Model Lifecycle

> Beyond just training — this lab demonstrates the full deep learning model lifecycle: training, TensorBoard logging, model saving, reloading, and parameter verification on the Fashion-MNIST dataset.

---

## 📌 Project Overview

This project moves past the "train and evaluate" stage and shows what happens after a model is trained. Using the Fashion-MNIST dataset of clothing images, it covers the complete lifecycle of a Keras model — from training and logging to serialization, reload, and structural verification.

A critical skill for any production-ready deep learning workflow.

---

## 📊 Dataset at a Glance

| Property | Value |
|---|---|
| Dataset | Fashion-MNIST |
| Classes | 10 (T-shirt, Trouser, Pullover, Dress, Coat, etc.) |
| Input Shape | 28×28 grayscale images |
| Train / Test Split | 60,000 / 10,000 |

---

## 🔍 What This Lab Demonstrates

| Concept | Implementation |
|---|---|
| Data loading | Fashion-MNIST via Keras datasets |
| Sample visualization | Matplotlib image grid |
| Preprocessing | Normalization + one-hot encoding |
| Model training | Dense NN with validation monitoring |
| Experiment logging | TensorBoard callback |
| Model serialization | Save with Keras native `.keras` format |
| Model reloading | Load and run inference |
| Structural verification | Parameter count comparison before/after reload |

---

## 🏗️ Model Architecture

```text
Flatten(28 × 28)
    ↓
Dense(128, ReLU)
    ↓
Dense(64, ReLU)
    ↓
Dense(10, Softmax)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | Model building, training, saving |
| TensorBoard | Training metric visualization |
| NumPy | Data manipulation |
| Matplotlib | Sample image visualization |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/sainathac/fashion-mnist-model-lifecycle.git
cd fashion-mnist-model-lifecycle
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the lab
```bash
python src/main.py
```

### 4. View TensorBoard logs (optional)
```bash
tensorboard --logdir=logs/
```

---

## 📚 Part of the Deep Learning Lab Series

This is **Lab 3 of 6** in a TensorFlow/Keras deep learning learning series:

1. [Neural Network From Scratch](https://github.com/sainathac/neural-network-from-scratch)
2. [MNIST Digit Classifier](https://github.com/sainathac/mnist-digit-classifier)
3. **[Fashion-MNIST Model Lifecycle](https://github.com/sainathac/fashion-mnist-model-lifecycle)** ← You are here
4. [Cats vs Dogs CNN Classifier](https://github.com/sainathac/cats-vs-dogs-cnn-classifier)
5. [VGG16 Transfer Learning Classifier](https://github.com/sainathac/vgg16-transfer-learning-classifier)
6. [IMDB Sentiment RNN](https://github.com/sainathac/imdb-sentiment-rnn)

---

## 👤 Author

**Sainatha C**
AI Automation & RPA Engineer | Data Science Practitioner

[![GitHub](https://img.shields.io/badge/GitHub-sainathac-181717?logo=github)](https://github.com/sainathac)

