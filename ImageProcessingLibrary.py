"""
DSC 20 Project
Name(s): Nathan Dang, Kevin Wu
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        >>> RGBImage([[[255, 255, 256], [0, 0, 0]]])
        Traceback (most recent call last):
        ...
        ValueError:s Pixel values must be in the range 0-255
        """
        
        if not isinstance(pixels, list) or not pixels:
            raise TypeError()
        if not all(len(row) == len(pixels[0]) for row in pixels):
            raise TypeError()

        for row in pixels:
            if not isinstance(row, list) or not row:
                raise TypeError()
            for row in pixels:
                for pixel in row:
                    if not all(0 <= val <= 255 for val in pixel):
                        raise ValueError("Pixel values must be in the range 0-255")


        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """

        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        
        return [[[pix_val for pix_val in col] for col in row] \
                    for row in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """

        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError
        >>> img.get_pixel(-1, -2)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """

        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if not row < self.num_rows or not col < self.num_cols:
            raise ValueError()
        if not row >= 0 or not col >= 0:
            raise ValueError()


        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        >>> img.set_pixel(0, 1, (100, 200, -1))
        >>> img.pixels
        [[[255, 0, 0], [100, 200, 0]]]
        """

        if not all(color <= 255 for color in new_color):
            raise ValueError()
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if not row < self.num_rows or not col < self.num_cols:
            raise ValueError()


        colors = list(self.get_pixel(row, col))
        for index, color in enumerate(new_color):
            if color < 0:
                continue
            else:
                colors[index] = color
        self.pixels[row][col] = colors
        


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """

        return self.cost


    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
          
        invert = map(lambda x: [[255 - val for val in vlst ] for vlst in x] , \
            image.pixels
        )


        return RGBImage(list(invert))

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """

        gray = [[[sum(val)//3]*3 for val in vlist] for vlist in image.pixels]

        
        return RGBImage(gray)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """

        rotate = [val[::-1] for val in image.pixels[::-1]]

        
        return RGBImage(rotate)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        >>> pixels = [[[0, 0, 0], [255, 255, 255]]]
        >>> img = RGBImage(pixels)
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.get_average_brightness(img)
        127
        """

        total_brightness = sum(
        sum(sum(pixel) // 3 for pixel in row) for row in image.pixels
        )
        avg_brightness = total_brightness // (image.num_rows * image.num_cols)

        return avg_brightness

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """

        brightness = map(lambda x: [[val + intensity for val in vlst ]    
            for vlst in x], image.pixels) 


        adjusted = [
            [
                [max(0, min(255, val + intensity)) for val in pixel]
            for pixel in row
            ]
            for row in image.pixels
        ]
        
        return RGBImage(adjusted)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        
        rows = image.num_rows
        cols = image.num_cols
        blurred_pixels = [[[0, 0, 0] for i in range(cols)] for i in range(rows)]
        for row in range(rows):
            for col in range(cols):
                pixels = image.pixels
                total = [0,0,0]
                count = 0

                for row_dir in [-1, 0, 1]:
                    for col_dir in [-1, 0, 1]:
                        r, c = row + row_dir, col + col_dir
                        if 0 <= r < rows and 0 <= c < cols:
                            total[0] += pixels[r][c][0]
                            total[1] += pixels[r][c][1]
                            total[2] += pixels[r][c][2]
                            count += 1
                blurred_pixels[row][col] = [total[0] // count, total[1] // count, total[2] // count]

        return RGBImage(blurred_pixels)

# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        super().__init__()
        self.cost = 0
        self.coupon_total = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """

        if self.coupon_total > 0:
            self.coupon_total -= 1
            self.cost +=0
        else:
            self.cost += 5
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image
        """

        if self.coupon_total > 0:
            self.coupon_total -= 1
            self.cost +=0
        else:
            self.cost +=6
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """

        if self.coupon_total > 0:
            self.coupon_total -= 1
            self.cost +=0
        else:
            self.cost += 10
        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """

        if self.coupon_total > 0:
            self.coupon_total -= 1
            self.cost +=0
        else:
            self.cost += 1
        return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """

        if self.coupon_total > 0:
            self.coupon_total -= 1
            self.cost +=0
        else:
            self.cost += 5

        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """

        if not isinstance(amount, int):
            raise TypeError()
        elif amount <= 0:
            raise ValueError()

        self.coupon_total += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """

        super().__init__()
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)
        """

        if not isinstance(chroma_image, RGBImage):
            raise TypeError()
        if not isinstance(background_image, RGBImage):
            raise TypeError()
        if chroma_image.size() != background_image.size():
            raise ValueError()

        background_pixels = background_image.pixels
        chroma_pixels = chroma_image.pixels

        rows = background_image.num_rows
        cols = background_image.num_cols

        for row in range(rows):
            for col in range(cols):
                if chroma_pixels[row][col] == list(color):
                    chroma_pixels[row][col] = background_pixels[row][col]

        return RGBImage(chroma_pixels)

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """

        rows = sticker_image.num_rows
        cols = sticker_image.num_cols

        if not isinstance(sticker_image, RGBImage):
            raise TypeError()
        if not isinstance(background_image, RGBImage):
            raise TypeError()

        if (
            rows > background_image.num_rows or 
            cols > background_image.num_cols
        ):
            raise ValueError()

        if not isinstance(x_pos, int):
            raise TypeError()
        if not isinstance(y_pos, int):
            raise TypeError()


        
        if cols + x_pos > background_image.num_cols or rows + y_pos > background_image.num_rows:
            raise ValueError()



        x, y = cols, rows


        for row in range(y):
            for col in range(x):
                actual_x = x_pos + col
                actual_y = y_pos + row
                background_image.pixels[actual_y][actual_x] = sticker_image.pixels[row][col]

        return RGBImage(background_image.pixels)



    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """

        rows = image.num_rows
        cols = image.num_cols

        # Grayscale conversion (outside the main loop)
        grayscale = [[sum(pixel) // 3 for pixel in row] for row in image.pixels]

        # Kernel for edge detection
        kernel = [
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]

        # Placeholder for the highlighted image
        black_pixels = [[[0, 0, 0] for _ in range(cols)] for _ in range(rows)]

        # Apply the kernel to each pixel
        for row in range(rows):
            for col in range(cols):
                total = 0
                for k_row in range(-1, 2):
                    for k_col in range(-1, 2):
                        # Neighbor coordinates
                        n_row = row + k_row
                        n_col = col + k_col

                        # Check boundaries
                        if 0 <= n_row < rows and 0 <= n_col < cols:
                            kernel_value = kernel[k_row + 1][k_col + 1]
                            total += grayscale[n_row][n_col] * kernel_value

                # Clamp the result to [0, 255]
                clamped_value = max(0, min(255, total))

                # Update all channels with the clamped value
                black_pixels[row][col] = [clamped_value] * 3

        return RGBImage(black_pixels)
             

# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        self.k_neighbors = k_neighbors

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        if len(data) < self.k_neighbors:
            raise ValueError()
        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance("not_an_image", "another_string")
        Traceback (most recent call last):
        ...
        TypeError: Both inputs must be instances of RGBImage
        """

        if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError("Both inputs must be instances of RGBImage")

        size1 = image1.size()
        size2 = image2.size()

        if size1 != size2:
            raise ValueError("Image dimensions must match")

        img1_pixels = [pixel for row in image1.pixels for pixel in row]
        img2_pixels = [pixel for row in image2.pixels for pixel in row]

        distance = sum(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
            for p1, p2 in zip(img1_pixels, img2_pixels)
        )

        return distance ** 0.5


    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        >>> knn.vote(['label1', 'label2', 'label2', 'label1'])
        'label1'
        >>> knn.vote(['aaaa', 'label2', 'label1'])
        'aaaa'
        """

        mostf = candidates[0]
        count = 0
        for label in candidates:
            labelcount = candidates.count(label)
            if labelcount > count:
                count = labelcount
                mostf = label 
        return mostf
        

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        
        try:
            self.data
        except NameError:
            raise ValueError()

        diff_calculated = [[self.distance(image, x[0])] + [x[1]] for x in self.data]
        


        sorted_diff = sorted([(item[0], item[1]) for item in diff_calculated])[:self.k_neighbors]

        labels = [x[1] for x in sorted_diff]

        return self.vote(labels)

def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'

    """

    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)

    return predicted_label
