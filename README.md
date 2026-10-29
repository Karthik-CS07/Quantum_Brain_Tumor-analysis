# Quantum Brain Tumor Analysis 🧠⚛️

A Hybrid Quantum-Classical Deep Learning framework for classifying brain tumors from MRI scans using **PennyLane** and **TensorFlow / Keras**.

---

## 📌 Overview

This project implements a hybrid quantum-classical neural network (HQNN) prototype designed to assist in medical image diagnosis. By combining convolutional layers for visual feature extraction with parameterized variational quantum circuits (VQCs), the model leverages quantum entanglement and superposition to enhance multi-class classification of brain tumors.

### Target Classes:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor (Healthy)**

---

## 🏗️ Architecture

```
MRI Image (64x64x3)
        │
        ▼
Classical CNN Feature Extractor (Conv2D + MaxPooling + Dense)
        │
        ▼ (4 features)
Angle Embedding (Pauli-Y rotations on 4 Qubits)
        │
        ▼
Parameterized Variational Quantum Circuit (PennyLane QNode)
        │
        ▼ (Expectation values <Z>)
Classical Classification Head (Softmax / 4 Classes)
```

1. **Classical Backbone**: Convolutional feature extractor reduces high-dimensional MRI image representations into a compact 4-dimensional latent vector.
2. **Quantum Layer**:
   - Implemented via **PennyLane** (`default.qubit` simulator).
   - Angle embedding encodes classical features using Pauli-Y rotations across 4 qubits.
   - Variational entanglement layers with trainable parameter weights.
   - Pauli-Z expectation value measurements.
3. **Classification Head**: Dense output layer with Softmax activation predicting class probabilities.

---

## 📂 Repository Structure

```
.
├── .gitignore                          # Ignored temporary & cache files
├── README.md                           # Project documentation
└── quantum-brain-tumor-prototype.ipynb # End-to-end training and evaluation notebook
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed.

```bash
pip install tensorflow pennylane numpy pandas matplotlib kagglehub
```

### Dataset

The model is trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar.

- When running in Kaggle, the notebook automatically discovers dataset paths under `/kaggle/input/`.
- For local runs, download the dataset and point the dataset directory variables in the notebook to your local training/testing paths.

### Running the Notebook

Open and run the notebook in Jupyter Lab, VS Code, or upload it to Kaggle / Google Colab:

```bash
jupyter notebook quantum-brain-tumor-prototype.ipynb
```

---

## 📈 Features & Optimization

- **Learning Rate Scheduling**: `ReduceLROnPlateau` dynamically halves the learning rate when validation accuracy plateaus.
- **Early Stopping**: Prevents overfitting and restores the best model weights.
- **Rapid Prototyping Pipeline**: Includes quick validation steps to verify quantum circuit integration and gradient flow before launching full-scale training.

---

## 👤 Author

- **Karthik** ([@Karthik-CS07](https://github.com/Karthik-CS07))
