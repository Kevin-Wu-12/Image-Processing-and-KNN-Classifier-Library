<h1 class="custom-title">Image Processing and KNN Classifier Library</h1>

A Python library for image processing and classification, built primarily with native Python functions and minimal dependencies. Features include customizable image manipulations (grayscale, negation, edge detection) and a K-Nearest Neighbors (KNN) classifier for image recognition, utilizing NumPy and PIL for performance optimization.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Running Tests](#running-tests)
5. [Contributions](#contributions)
6. [Disclaimer](#disclaimer)

## **Overview**
This Python library provides a modular framework for advanced image processing and classification using a K-Nearest Neighbors (KNN) approach. It includes classes and methods for manipulating RGB images, implementing standard and premium image processing techniques, and classifying images based on pixel similarity.


## **Contributions**
This project focuses on the implementation of:
- RGBImage class for handling pixel-based image manipulation.
- ImageProcessingTemplate and derived classes for standard and premium image processing.
- ImageKNNClassifier for KNN-based image classification.

The `image_viewer.py` file was provided as part of the starter code for this project. It is used for testing and demonstrating the functionality of the implemented library but was not
written by me.

## Running Tests
This project uses doctests for validating functionality. To run the doctests:

```bash
python -m doctest -v ImageProcessingLibrary.py
```

### Disclaimer
This library is a personal implementation primarily built using native Python. While it emphasizes simplicity, clarity, and educational value, it may not be as optimized or fast as other specialized libraries like OpenCV or scikit-image. It is best suited for learning purposes or smaller-scale applications.

# Project Documentation

---

## Features

### 1. RGB Image Manipulation
- **`RGBImage` Class**:
  - Represents an image as a matrix of RGB values.
  - Allows pixel-level manipulations.
  - Provides utility functions like:
    - `get_pixel()`: Retrieve the RGB value of a specific pixel.
    - `set_pixel()`: Modify the RGB value of a pixel.
    - `size()`: Returns the dimensions of the image.

### 2. Image Processing
- **StandardImageProcessing**:
  - Basic image manipulations like: 
    - `negate`: Replace specific colors with a background.
    - `grayscale`: Overlay one image on another at specified positions.
    - `rotate_180`: Highlight edges in an image.
    - `adjust_brightness`: Highlight edges in an image.
    - `blur`: Highlight edges in an image.

- **PremiumImageProcessing**:
  - Advanced features like:
    - `chroma_key`: Replace specific colors with a background.
    - `sticker`: Overlay one image on another at specified positions.
    - `edge_highlight`: Highlight edges in an image.

### 3. Image Classification
- **ImageKNNClassifier**:
  - Implements a K-Nearest Neighbors classifier for image labels.
  - Features:
    - `fit()`: Train the classifier on labeled image data.
    - `predict()`: Classify an image based on its nearest neighbors.
    - `distance()`: Compute the Euclidean distance between two images.

### Prerequisites
- Python 3.x
- Required Libraries: `numpy`, `PIL`, `os`
# ImageKNNClassifier

## Overview
A simple K-Nearest Neighbors (KNN) classifier for image data, supporting Euclidean distance-based classification.

---

## Key Methods

### `__init__(self, k_neighbors)`
Initializes the classifier.

- **Parameters**:
  - `k_neighbors` (int): Number of neighbors to consider for classification.

---

### `fit(self, data)`
Trains the classifier using labeled image data.

- **Parameters**:
  - `data` (list of tuples): Each tuple contains an `RGBImage` instance and its associated label.

- **Raises**:
  - `ValueError`: If the dataset has fewer elements than `k_neighbors`.

---

### `predict(self, image)`
Predicts the label of an input image based on its nearest neighbors.

- **Parameters**:
  - `image` (`RGBImage`): The image to classify.

- **Returns**:
  - `str`: Predicted label.

- **Raises**:
  - `ValueError`: If the classifier has not been trained.

---

### `distance(self, image1, image2)`
Calculates the Euclidean distance between two images.

- **Parameters**:
  - `image1`, `image2` (`RGBImage`): The two images to compare.

- **Returns**:
  - `float`: The computed Euclidean distance.

- **Raises**:
  - `TypeError`: If inputs are not `RGBImage` instances.
  - `ValueError`: If the dimensions of the two images do not match.


## **Installation**
Clone the repository and ensure you have the following dependencies installed:
```bash
git clone https://kevin-wu-12.github.io/Image-Processing-and-KNN-Classifier-Library/
cd Image-Processing-and-KNN-Classifier-Library
pip install -r requirements.txt

