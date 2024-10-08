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

## Introduction

This notebook focuses on building a comprehensive NLP pipeline for binary sentiment classification using movie reviews from the IMDB dataset. The notebook includes text preprocessing steps, feature extraction techniques (e.g., TF-IDF, Bag of Words, Word2Vec, and GloVe), and model training using classifiers such as Logistic Regression, XGBoost, and deep learning models.

## Dataset

The dataset used is the IMDB Dataset of 50K Movie Reviews, which consists of movie reviews labeled as positive or negative. It is available on Kaggle.

## Features

- Data Preprocessing: Includes tokenization, stop-word removal, stemming, and text cleaning.
- Feature Extraction: Implements TF-IDF, Bag of Words, Word2Vec, and GloVe embeddings.
- Modeling: Trains models using Logistic Regression, XGBoost, and other classifiers.
- Performance Evaluation: Evaluates models using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.
- Hyperparameter Tuning: Uses GridSearchCV and RandomizedSearchCV to optimize model performance.

## Models

- Logistic Regression: For both Bag of Words and TF-IDF features.
- XGBoost Classifier: Implemented with various feature extraction techniques (TF-IDF, Bag of Words, Word2Vec, and GloVe).
- Deep Learning (RNNs/CNNs): Using GloVe embeddings and PyTorch or TensorFlow for model building.

## Dependencies

The following libraries are required to run the notebook:

- `numpy`: For numerical computations.
- `pandas`: For data manipulation.
- `matplotlib`: For data visualization.
- `seaborn`: For enhanced visualizations.
- `sklearn`: For machine learning models and evaluation metrics.
- `xgboost`: For gradient boosting classifier.
- `nltk`: For text preprocessing (tokenization, stop-word removal).
- `torch`: For deep learning models using PyTorch.
- `tensorflow`: For alternative deep learning models and text preprocessing.
  
You can install the dependencies using the following command:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost nltk torch tensorflow
```
## Results

The notebook outputs several evaluation metrics, including accuracy, precision, recall, and F1-score. Confusion matrices are visualized to provide a clearer understanding of model performance on the IMDB dataset.

- Best Accuracy: 88.21% using tuned BiLSTM with GloVe Embeddings.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.
