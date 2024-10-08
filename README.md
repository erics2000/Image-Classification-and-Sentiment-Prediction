# Overview

This repository includes two Python notebooks: one focuses on image classification using the CIFAR10 dataset, while the other performs sentiment analysis on IMDB reviews.

# Image Classification using CIFAR10 Dataset

## Introduction

This project focuses on training and evaluating models for image classification tasks using the CIFAR10 dataset. It includes preprocessing steps such as normalization and PCA, and builds models such as a Softmax classifier and two Convolutional Neural Networks (CNNs). The notebook also explores hyperparameter tuning and model evaluation using various performance metrics.

## Dataset

The CIFAR-10 dataset can be downloaded from the official website. It is split into a training set of 50,000 images and a test set of 10,000 images. Each image is a 32x32 pixel RGB image, with 3 color channels.

## Features

- Data Preprocessing: Standardization, dimensionality reduction with PCA, and dataset splitting.
- Softmax Classifier: Implements a basic Softmax classifier with hyperparameter tuning.
- CNN Architectures: Two CNN models (StudentCNN1 and StudentCNN2) with dropout and batch normalization for regularization.
- Model Evaluation: Training, validation, and test accuracy evaluation, with confusion matrix visualization.

## Models

- Softmax Classifier: A simple classifier with hyperparameter tuning, using gradient descent optimization.
- CNN1 (StudentCNN1): A CNN with three convolutional layers followed by fully connected layers. Trained using SGD.
- CNN2 (StudentCNN2): A deeper CNN with batch normalization and dropout layers for regularization.

## Dependencies

The following libraries are required to run the notebook:

- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For data visualization.
- `seaborn`: For enhanced data visualizations.
- `scikit-learn`: For machine learning tasks like data preprocessing and evaluation.
- `torch` (PyTorch): For building and training neural networks.
- `torchvision`: For image transformations and loading the CIFAR10 dataset.

To install these dependencies, run the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision
```

## Result

The notebook includes visualizations for training and validation accuracy, loss curves, and confusion matrices. CNN2 achieves an accuracy of 82.01% on the CIFAR10 test set, showing improvement over CNN1.

# Sentiment Analysis of IMDB Review

- **Image Preprocessing**: Handles normalization, resizing, and augmentation of images.
- **Model Training**: Supports different models from popular deep learning frameworks.
- **Evaluation Metrics**: Computes accuracy, loss, and other relevant metrics for model performance.
- **Prediction and Visualization**: Includes methods for model inference and visualization of results.

## Installation

To run the notebook, you will need to clone this repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/CV_Pipeline.git
cd CV_Pipeline
