# Fashion-MNIST Model Lifecycle

A Fashion-MNIST classifier demonstrating model training, visualization, TensorBoard logging, model saving, loading, and parameter verification.

## Overview

This lab moves beyond simple model training and demonstrates a basic deep learning model lifecycle. It trains a dense neural network on Fashion-MNIST, logs metrics to TensorBoard, saves the trained model, reloads it, and verifies that the model structure is preserved.

## What This Lab Demonstrates

- Fashion-MNIST dataset loading
- Sample image visualization
- Data normalization and one-hot encoding
- Dense neural network training
- Validation monitoring
- TensorBoard callback setup
- Model saving with Keras native format
- Model loading and parameter-count verification

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the lab:

```bash
python src/main.py
```

Open TensorBoard logs:

```bash
tensorboard --logdir logs
```

## Dataset

Fashion-MNIST is loaded directly from `tensorflow.keras.datasets`. Generated logs and models are intentionally ignored by Git.
