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
7. [Documentation](#project-documentation)
   - [Image Helpers](#image-reader)
   - [Class `RGBImage`](#class-rgbimage)
   - [Class `StandardImageProcessing`](#class-standardimageprocessing)
   - [Class `PremiumImageProcessing`](#class-premiumimageprocessing)





## **Overview**
This Python library provides a modular framework for advanced image processing and classification using a K-Nearest Neighbors (KNN) approach. It includes classes and methods for manipulating RGB images, implementing standard and premium image processing techniques, and classifying images based on pixel similarity.


## **Contributions**
This project focuses on the implementation of:
- RGBImage class for handling pixel-based image manipulation.
- ImageProcessingTemplate and derived classes for standard and premium image processing.
- ImageKNNClassifier for KNN-based image classification.

The `image_viewer.py` file was provided as part of the starter code for this project. It is used for testing and demonstrating the functionality of the implemented library but was not
written by me.

### Prerequisites
- Python 3.x
- Required Libraries: `numpy`, `PIL`, `os`

## **Installation**
Clone the repository and ensure you have the following dependencies installed:
```bash
git clone https://kevin-wu-12.github.io/Image-Processing-and-KNN-Classifier-Library/
cd Image-Processing-and-KNN-Classifier-Library
pip install -r requirements.txt
```

## Running Tests
This project uses doctests for validating functionality. To run the doctests:

```bash
python -m doctest -v ImageProcessingLibrary.py
```

### Disclaimer
This library is a personal implementation primarily built using native Python. While it emphasizes simplicity, clarity, and educational value, it may not be as optimized or fast as other specialized libraries like OpenCV or scikit-image. It is best suited for learning purposes or smaller-scale applications.

## Examples

Explore the following example scripts to see the library in action:

1. [Grayscale Example](examples/grayscale_example.py): Convert an image to grayscale.
2. [KNN Classifier Example](examples/knn_classifier_example.py): Classify images using K-Nearest Neighbors.

### Sample Images
To use these examples, download the repository and navigate to the `sample_images/` folder for input images.

### Run an Example
Run the examples by navigating to the `examples/` folder and executing the script:

```bash
python grayscale_example.py
```
# Project Documentation

## Features

### 1. RGB Image Manipulation
- **`RGBImage` Class**:
  - Represents an image as a matrix of RGB values.
  - Allows pixel-level manipulations.
  - Provides utility functions like:
    - `get_pixel()`: Retrieve the RGB value of a specific pixel.
    - `set_pixel()`: Modify the RGB value of a pixel.
    - `size()`: Returns the dimensions of the image.
    - `copy()`': Returns a copy of this RGBImage object

### 2. Image Processing
- **StandardImageProcessing**:
  - Basic image manipulations like: 
    - `negate`: Replace specific colors with a background.
    - `grayscale`: Overlay one image on another at specified positions.
    - `rotate_180`: Highlight edges in an image.
    - `adjust_brightness`: Adjust brightness of an image.
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
    
# Image Reader 

## `img_read_helper(path)`
**Purpose**:  
Reads an image file from the given path, converts it into an RGB format, and creates an `RGBImage` object.

**Inputs**:  
- `path` (str): The file path to the image.

**Outputs**:  
- An instance of `RGBImage` created from the image.

**Exceptions Raised**:  
- Exceptions related to invalid file paths or unreadable image files.

**Remarks**:  
- The image is always converted to RGB format regardless of its original mode.

---

## `img_save_helper(path, image)`
**Purpose**:  
Saves an `RGBImage` instance as an image file at the specified path.

**Inputs**:  
- `path` (str): The file path where the image will be saved.
- `image` (`RGBImage`): An instance of the `RGBImage` class to save.

**Outputs**:  
- None.

**Exceptions Raised**:  
- None.

---

# Class `RGBImage`

### `__init__(pixels)`
**Purpose**:  
Initializes an `RGBImage` instance with a 3D list of pixel values.

**Inputs**:  
- `pixels` (list): A 3D list where each element represents the RGB values of a pixel.

**Outputs**:  
- None.

**Exceptions Raised**:  
- `TypeError`: If the input is not a rectangular 3D list.
- `ValueError`: If pixel values are not integers in the range [0, 255].

---

### `size()`
**Purpose**:  
Returns the dimensions of the image.

**Inputs**:  
- None.

**Outputs**:  
- A tuple `(num_rows, num_cols)` representing the number of rows and columns in the image.

---

### `get_pixels()`
**Purpose**:  
Returns a deep copy of the image's pixel data.

**Inputs**:  
- None.

**Outputs**:  
- A 3D list of the image's pixel data.

---

### `copy()`
**Purpose**:  
Returns a copy of the `RGBImage` instance

**Inputs**:  
- None.

**Outputs**:  
- A `RGBImage` instance

**Exceptions Raised**:  
- None.

---

### `get_pixel()`
**Purpose**:  
Returns the (R, G, B) value at the given positio

**Inputs**:  
- None.

**Outputs**:  
- A 3D list of the image's pixel data.

---

### `set_pixel(row, col, new_color)`
**Purpose**:  
Updates the RGB value of a specific pixel.

**Inputs**:  
- `row` (int): The row index.
- `col` (int): The column index.
- `new_color` (tuple): A tuple `(R, G, B)` with the new pixel values. Negative values are ignored.

**Outputs**:  
- None.

**Exceptions Raised**:  
- `TypeError`: If `row` or `col` is not an integer.
- `ValueError`: If the RGB values in `new_color` exceed the range [0, 255].

---

# Class `StandardImageProcessing`

### `__init__()`
**Purpose**:  
Initializes an `ImageProcessingTemplate` instance with the cost and coupon total of 0

**Inputs**:  
- None.

**Outputs**:  
- None.

---

### `negate(image)`
**Purpose**:  
Creates a new image with the RGB values negated (inverted).

**Inputs**:  
- `image` (`RGBImage`): The input image.

**Outputs**:  
- A new `RGBImage` with negated pixel values.

---

### `grayscale(image)`
**Purpose**:  
Creates a grayscale version of the given image.

**Inputs**:  
- `image` (`RGBImage`): The input image.

**Outputs**:  
- A new `RGBImage` in grayscale.

---

### `rotate_180(image)`
**Purpose**:  
Creates a new image rotated 180 degrees.

**Inputs**:  
- `image` (`RGBImage`): The input image.

**Outputs**:  
- A new `RGBImage` rotated by 180 degrees.

---

### `get_average_brightness(image)`
**Purpose**:  
Calculates the average brightness of the image.

**Inputs**:  
- `image` (`RGBImage`): The input image.

**Outputs**:  
- An integer representing the average brightness.

---

### `adjust_brightness(image, intensity)`
**Purpose**:  
Adjusts the brightness of the image by the given intensity.

**Inputs**:  
- `image` (`RGBImage`): The input image.
- `intensity` (int): The adjustment value (positive or negative).

**Outputs**:  
- A new `RGBImage` with adjusted brightness.

---

### `blur(image)`
**Purpose**:  
Applies a blur effect to the image.

**Inputs**:  
- `image` (`RGBImage`): The input image.

**Outputs**:  
- A new `RGBImage` with a blur effect applied.

---

# Class `PremiumImageProcessing`

Represents a paid tier of an image processor, extending the `ImageProcessingTemplate` class.

### `__init__()`
**Purpose**:  
Initializes a `PremiumImageProcessing` object with an initial cost of 50.

**Inputs**:  
- None.

**Outputs**:  
- None.

**Exceptions Raised**:  
- None.

---

## `chroma_key(chroma_image, background_image, color)`

**Purpose:**  
Replaces all pixels in the `chroma_image` that match the specified `color` with corresponding pixels from the `background_image`.

**Inputs:**  
- `chroma_image (RGBImage)`: The image with the chroma key to be replaced.  
- `background_image (RGBImage)`: The background image used for replacement.  
- `color (tuple)`: A tuple `(R, G, B)` representing the chroma key color.

**Outputs:**  
- A new `RGBImage` with the chroma key applied.

**Exceptions Raised:**  
- `TypeError`: If `chroma_image` or `background_image` is not an `RGBImage`.  
- `ValueError`: If the dimensions of `chroma_image` and `background_image` do not match.

---

## `sticker(sticker_image, background_image, x_pos, y_pos)`

**Purpose:**  
Places the `sticker_image` on top of the `background_image` at the specified position (`x_pos`, `y_pos`).

**Inputs:**  
- `sticker_image (RGBImage)`: The image to place as a sticker.  
- `background_image (RGBImage)`: The background image.  
- `x_pos (int)`: The x-coordinate of the sticker's top-left corner.  
- `y_pos (int)`: The y-coordinate of the sticker's top-left corner.

**Outputs:**  
- A new `RGBImage` with the sticker applied.

**Exceptions Raised:**  
- `TypeError`: If `sticker_image` or `background_image` is not an `RGBImage`, or if `x_pos` or `y_pos` is not an integer.  
- `ValueError`: If the sticker goes out of bounds of the background image.

---

## `edge_highlight(image)`

**Purpose:**  
Highlights the edges in the image using an edge detection kernel.

**Inputs:**  
- `image (RGBImage)`: The input image.

**Outputs:**  
- A new `RGBImage` with highlighted edges.

**Exceptions Raised:**  
- None.


---

## Class `ImageKNNClassifier`

### `__init__(k_neighbors)`
**Purpose**:  
Initializes a KNN classifier with a specified number of neighbors.

**Inputs**:  
- `k_neighbors` (int): The number of neighbors for classification.

---

### `fit(data)`
**Purpose**:  
Stores training data for the classifier.

**Inputs**:  
- `data` (list): A list of tuples `(image, label)`.

**Exceptions Raised**:  
- `ValueError`: If the number of training samples is less than `k_neighbors`.

---

### `distance(image1, image2)`
**Purpose**:  
Calculates the Euclidean distance between two images.

**Inputs**:  
- `image1` (`RGBImage`): The first image.
- `image2` (`RGBImage`): The second image.

**Outputs**:  
- A float representing the Euclidean distance.

**Exceptions Raised**:  
- `TypeError`: If either input is not an `RGBImage`.
- `ValueError`: If the image dimensions do not match.

---

### `vote(candidates)`
**Purpose**:  
Determines the most frequent label among the given candidates.

**Inputs**:  
- `candidates` (list): A list of labels.

**Outputs**:  
- The label with the highest frequency.

---

### `predict(image)`
**Purpose**:  
Predicts the label of the given image based on the k-nearest neighbors.

**Inputs**:  
- `image` (`RGBImage`): The input image.

**Outputs**:  
- The predicted label.

**Exceptions Raised**:  
- `ValueError`: If the classifier has not been trained.

---

## `knn_tests(test_img_path)`

**Purpose:**  
Runs K-Nearest Neighbors (KNN) classification on an image, predicting its label based on a dataset of labeled images.

**Inputs:**  
- `test_img_path (str)`: The file path of the test image to classify.

**Outputs:**  
- `str`: The predicted label for the test image.

**Exceptions Raised:**  
- `FileNotFoundError`: If the provided `test_img_path` does not exist or cannot be read.  
- `ValueError`: If the dataset folder (`knn_data`) is empty or improperly structured.
