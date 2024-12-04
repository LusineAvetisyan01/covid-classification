# COVID-19 Detection Using Deep Learning

This project utilizes deep learning for detecting COVID-19 cases based on chest X-ray images. The model is built using PyTorch and ResNet-18 architecture. The system performs binary classification to classify images as either `COVID` or `non-COVID`.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation Instructions](#installation-instructions)
4. [Model Training](#model-training)
5. [Inference & Evaluation](#inference--evaluation)
6. [Metrics](#metrics)
7. [License](#license)

---

## Project Overview

The goal of this project is to create a deep learning model that can accurately classify chest X-ray images into two categories:
- **COVID**: The image shows signs of COVID-19 infection.
- **Non-COVID**: The image does not show any signs of COVID-19.

The model was trained using a dataset containing labeled chest X-ray images. A ResNet-18 architecture was chosen for its proven performance in image classification tasks.

---

## Prerequisites

Before running this project, ensure that you have the following installed:

- Python 3.x
- PyTorch (with CUDA support if you're using a GPU)
- torchvision
- scikit-learn
- Pillow
- NumPy

You can install the necessary libraries using pip:

```bash
pip install torch torchvision scikit-learn Pillow numpy
