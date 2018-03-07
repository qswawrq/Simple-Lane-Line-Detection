# Shuwei Qiang
# March 4, 2018
# Project 1: Simple Lane Line Detection

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import os
import cv2

# Parameters for image processing
path = "resource/images/"
image_index = 1
kernel_size = 5
canny_low = 75
canny_high = 175
top_left_corner_percentage_x = 0.4
top_right_corner_percentage_x = 0.6
bottom_left_corner_percentage_x = 0.1
bottom_right_corner_percentage_x = 0.9
top_left_corner_percentage_y = 0.65
top_right_corner_percentage_y = 0.65
bottom_left_corner_percentage_y = 1
bottom_right_corner_percentage_y = 1

# Grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur
def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Canny transform
def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def generate_mask(image):
    dimension = image.shape
    top_left = [math.floor(top_left_corner_percentage_x * dimension[1]), math.floor(top_left_corner_percentage_y * dimension[0])]
    top_right = [math.floor(top_right_corner_percentage_x * dimension[1]), math.floor(top_right_corner_percentage_y * dimension[0])]
    bottom_left = [math.floor(bottom_left_corner_percentage_x * dimension[1]), math.floor(bottom_left_corner_percentage_y * dimension[0])]
    bottom_right = [math.floor(bottom_right_corner_percentage_x * dimension[1]), math.floor(bottom_right_corner_percentage_y * dimension[0])]
    shape = np.array([top_left, top_right, bottom_left, bottom_right])
    return shape

# Read a list of images
images = os.listdir(path)
# Choose the target image
image_original = mpimg.imread(path + images[image_index]);
# Process the image
image_gray = grayscale(image_original)
image_blur = gaussian_blur(image_gray, kernel_size);
image_canny = canny(image_blur, canny_low, canny_high)
generate_mask(image_canny)
# Plot the image in gray scale and show the image
imgplot = plt.imshow(image_canny, cmap = 'gray')
plt.show()
