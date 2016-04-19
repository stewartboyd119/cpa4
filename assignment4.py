# ASSIGNMENT 4
# Stewart Boyd

import cv2
import numpy as np
import scipy as sp

""" Assignment 4 - Detecting Gradients / Edges

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file. (This is a problem
    for us when grading because running 200 files results a lot of images being
    saved to file and opened in dialogs, which is not ideal). Thanks.

    2. DO NOT import any other libraries aside from the three libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. Do not put functions into classes,
    or your own infrastructure. This makes grading very difficult for us. Please
    only write code in the allotted region.
"""

def imageGradientX(image):
    """ This function differentiates an image in the X direction.

    Note: See lectures 02-06 (Differentiating an image in X and Y) for a good
    explanation of how to perform this operation.

    The X direction means that you are subtracting columns:
    der. F(x, y) = F(x+1, y) - F(x, y)
    This corresponds to image[r,c] = image[r,c+1] - image[r,c]

    You should compute the absolute value of the differences in order to avoid
    setting a pixel to a negative value which would not make sense.

    We want you to iterate the image to complete this function. You may NOT use
    any functions that automatically do this for you.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the X direction. The shape
                                of the output array should have a width that is
                                one less than the original since no calculation
                                can be done once the last column is reached. 
    """
    # WRITE YOUR CODE HERE.

    (num_rows, num_cols) = image.shape
    new_image = np.zeros((num_rows, num_cols - 1))
    for j in xrange(num_cols - 1):
        for i in xrange(num_rows):
            delta = int(image[i, j + 1]) - int(image[i, j])
            new_image[i, j] = abs(delta)
    return new_image            


    # END OF FUNCTION.

def imageGradientY(image):
    """ This function differentiates an image in the Y direction.

    Note: See lectures 02-06 (Differentiating an image in X and Y) for a good
    explanation of how to perform this operation.

    The Y direction means that you are subtracting rows:
    der. F(x, y) = F(x, y+1) - F(x, y)
    This corresponds to image[r,c] = image[r+1,c] - image[r,c]

    You should compute the absolute value of the differences in order to avoid
    setting a pixel to a negative value which would not make sense.

    We want you to iterate the image to complete this function. You may NOT use
    any functions that automatically do this for you.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the Y direction. The shape
                                of the output array should have a height that is
                                one less than the original since no calculation
                                can be done once the last row is reached.
    """
    # WRITE YOUR CODE HERE.

    (num_rows, num_cols) = image.shape
    new_image = np.zeros((num_rows - 1, num_cols))
    for j in xrange(num_cols):
        for i in xrange(num_rows - 1):
            delta = int(image[i + 1, j]) - int(image[i, j])
            new_image[i, j] = abs(delta)
    return new_image            


    # END OF FUNCTION.

def computeGradient(image, kernel):
    """ This function applies an input 3x3 kernel to the image, and outputs the
    result. This is the first step in edge detection which we discussed in
    lecture.

    You may assume the kernel is a 3 x 3 matrix.
    View lectures 2-05, 2-06 and 2-07 to review this concept.

    The process is this: At each pixel, perform cross-correlation using the
    given kernel. Do this for every pixel, and return the output image.

    The most common question we get for this assignment is what do you do at
    image[i, j] when the kernel goes outside the bounds of the image. You are
    allowed to start iterating the image at image[1, 1] (instead of 0, 0) and
    end iterating at the width - 1, and column - 1.
    
    Note: The output is a gradient depending on what kernel is used.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.
        kernel (numpy.ndarray): A 3x3 kernel represented in a numpy array.

    Returns:
        output (numpy.ndarray): The computed gradient for the input image. The
                                size of the output array should be two rows and
                                two columns smaller than the original image
                                size.
    """                            
    # WRITE YOUR CODE HERE.

    (num_row, num_col) = image.shape
    new_image = np.zeros((num_row - 2, num_col - 2))
    for j in xrange(1, num_col - 1):
        for i in xrange(1, num_row - 1):
            new_image[i - 1, j - 1] = (kernel * image[i - 1:i + 2, j - 1: j + 2]).sum()
    return new_image
        


    # END OF FUNCTION.
    