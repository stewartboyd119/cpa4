import assignment4
import cv2
import numpy as np
import os

def convertToBlackAndWhite(image, threshold = 128):
    """ This function converts a grayscale image to black and white.

    Assignment Instructions: Iterate through every pixel in the image. If the
    pixel is strictly greater than 128, set the pixel to 255. Otherwise, set the
    pixel to 0. You are essentially converting the input into a 1-bit image, as 
    we discussed in lecture, it is a 2-color image.

    You may NOT use any thresholding functions provided by OpenCV to do this.
    All other functions are fair game.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        numpy.ndarray: The black and white image.
    """
    # WRITE YOUR CODE HERE.

    #---modified in place but returned to match api
    for elem in np.nditer(image, op_flags = ['readwrite']):
        elem[...] = 255 if elem > threshold else 0
    return image

def convert_and_write(image, image_name, threshold, outdir):
    im = convertToBlackAndWhite(image, threshold = threshold)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    cv2.imwrite(os.path.join(outdir, image_name.format(threshold)), im)

im = cv2.imread('test_image.jpg', 0)
test_thresholds = (50, 100, 128, 150, 200)
for index, threshold in enumerate(test_thresholds):
    print 'start threshold = {}'.format(threshold)
    #---pewitt
    kernel_pewitt = np.ndarray((3,3), buffer=np.array([[-1,-0,1], [-1, 0, 1], [-1, 0, 1]]), dtype=int)
    im3 = assignment4.computeGradient(im, kernel_pewitt)
    if index == 0:
        cv2.imwrite('prewitt.jpg', im3)
    convert_and_write(im3, 'prewitt_bw-{}.jpg', threshold, outdir = 'PEWITT')

    #---sobel
    kernel_sobelx = np.ndarray((3,3), buffer=np.array([[-1,-0,1], [-2, 0, 2], [-1, 0, 1]]), dtype=int)
    kernel_sobely = np.ndarray((3,3), buffer=np.array([[-1,-2, -1], [0, 0, 0], [1, 2, 1]]), dtype=int)
    
    im3x = assignment4.computeGradient(im, kernel_sobelx)
    im3y = assignment4.computeGradient(im, kernel_sobely)
    im3mag = np.sqrt(im3x**2 + im3y**2)
    if index == 0:
        cv2.imwrite('sobelx.jpg', im3x)
        cv2.imwrite('sobely.jpg', im3y)
        cv2.imwrite('sobelmag.jpg', im3mag)
    convert_and_write(im3x, 'sobelx_bw-{}.jpg', threshold, outdir = 'SOBELX')
    convert_and_write(im3y, 'sobely_bw-{}.jpg', threshold, outdir = 'SOBELY')
    convert_and_write(im3mag, 'sobelmag_bw-{}.jpg', threshold, outdir = 'SOBELMAG')

    #---roberts
    kernel_roberts = np.ndarray((3,3), buffer=np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]), dtype=int)
    im3 = assignment4.computeGradient(im, kernel_roberts)
    if index == 0:
        cv2.imwrite('roberts.jpg', im3)
    convert_and_write(im3, 'roberts_bw-{}.jpg', threshold, outdir = 'ROBERTS')

#---Compare to canny edge
canny_edge = cv2.Canny(im, 100, 200)
cv2.imwrite('canny_edge.jpg', canny_edge)
