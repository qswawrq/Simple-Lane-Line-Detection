# Shuwei Qiang
# March 4, 2018
# Project 1: Simple Lane Line Detection

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

# Grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur
def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Canny transform
def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

# Parameters for image processing
# TODO Put them in a parameter file
path = "resource/images/"
image_index = 1
kernel_size = 5
canny_low = 75
canny_high = 175

# Read a list of images
images = os.listdir(path)
# Choose the target image
image_original = mpimg.imread(path + images[image_index]);
# Process the image
image_gray = grayscale(image_original)
image_blur = gaussian_blur(image_gray, kernel_size);
image_canny = canny(image_blur, canny_low, canny_high)
# Plot the image in gray scale and show the image
imgplot = plt.imshow(image_canny, cmap = 'gray')
plt.show()

    

