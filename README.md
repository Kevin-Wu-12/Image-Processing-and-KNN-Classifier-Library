# Image-Processing-and-KNN-Classifier-Library
A Python library for image processing and classification, built primarily with native Python functions and minimal dependencies. 
Features include customizable image manipulations (grayscale, negation, edge detection) and a K-Nearest Neighbors (KNN) classifier for image recognition, 
utilizing NumPy and PIL for performance optimization.

## Running Tests
This project uses doctests for validating functionality. To run the doctests:

```bash
python -m doctest -v ImageProcessingLibrary.py
```
# **Image Processing and KNN Classifier Library**

## **Overview**
This Python library provides a modular framework for advanced image processing and classification using a K-Nearest Neighbors (KNN) approach. It includes classes and methods for manipulating RGB images, implementing standard and premium image processing techniques, and classifying images based on pixel similarity.

## **Features**
- **RGB Image Manipulation**:
  - Deep copying of image objects.
  - Access and modification of pixel data.
  - Retrieval of image dimensions and average brightness.
  
- **Standard Image Processing**:
  - Negate colors, grayscale, rotate, blur, and adjust brightness.
  
- **Premium Image Processing**:
  - Chroma keying for background replacement.
  - Sticker overlay functionality.
  - Edge highlighting using convolutional kernels.

- **KNN Classifier**:
  - Predicts image labels based on pixel similarity using K-Nearest Neighbors.
  - Customizable `k` parameter for fine-tuned classification.
  - Euclidean distance computation for image similarity.

## **Installation**
Clone the repository and ensure you have the following dependencies installed:
```bash
git clone https://github.com/yourusername/knn-image-library.git
cd knn-image-library
pip install -r requirements.txt
