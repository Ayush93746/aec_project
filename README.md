# Deep Learning-Based Step-Size Inference for Acoustic Echo Cancellation (AEC)

This repository contains a PyTorch implementation of a **Gated Recurrent Unit (GRU)-based Deep Neural Network** designed to enhance Acoustic Echo Cancellation by inferring optimal, frequency-dependent step sizes ($\mu$).

## 📌 Project Overview

In modern telecommunications, **Acoustic Echo Cancellation (AEC)** is essential for ensuring clear communication. Traditional adaptive filters like **LMS** or **Kalman filters** often struggle to balance convergence speed with stability. 

This project addresses these limitations by using a DNN to predict the ideal step size for each frequency bin in real-time, allowing for superior echo suppression even in dynamic acoustic environments.

## 🛠️ Methodology

- **Signal Processing:** Audio is processed in the frequency domain using Short-Time Fourier Transform (STFT).
- **Architecture:** A GRU-based network captures temporal dependencies in speech signals.
- **Optimization:** The model outputs a sigmoid-activated value ($\mu \in [0, 1]$) to scale the cancellation magnitude per frequency bin.
- **Evaluation:** Performance is measured using Echo Return Loss Enhancement (ERLE).

## 🚀 Key Features

* **Adaptive Step-Size Inference:** Real-time optimization of $\mu$.
* **Time-Series Modeling:** GRU layers to handle the sequential nature of audio.
* **HDF5 Integration:** Efficient data handling for large audio datasets.

---

## 🔍 Research Gaps & Future Work

As specified in the study, the following areas represent current limitations and opportunities for further development:

### 1. Computational Complexity
* **Description:** High-resolution DNNs require significant FLOPS, making real-time implementation on low-power mobile or IoT devices challenging without model quantization or pruning.

### 2. Dataset Diversity
* **Description:** Models trained on synthetic room impulse responses may struggle with real-world non-linearities caused by low-quality hardware or extreme reverberation.

### 3. Phase Reconstruction
* **Description:** Current magnitude-based masks often ignore phase information; integrating phase-aware processing could further reduce musical noise artifacts in the output.

---

## 📂 Project Structure

* `aec.ipynb`: The main notebook containing data loading, model architecture, and training loops.
* `data/`: Placeholder for `.h5` audio datasets.
* `models/`: Saved model weights and checkpoints.

## ⚙️ Requirements

- Python 3.x
- PyTorch
- H5py
- Librosa
- Matplotlib (for visualization)

---

Developed as a deep learning approach to signal processing.
