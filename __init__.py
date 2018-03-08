# Shuwei Qiang
# March 4, 2018
# Project 1: Simple Lane Line Detection

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import os
import cv2
from unittest.mock import right

# Parameters for image processing
path = "resource/images/"
image_index = 3
kernel_size = 5
canny_low = 75
canny_high = 175
top_left_corner_percentage_x = 0.4
top_right_corner_percentage_x = 0.6
bottom_left_corner_percentage_x = 0.08
bottom_right_corner_percentage_x = 0.92
top_left_corner_percentage_y = 0.65
top_right_corner_percentage_y = 0.65
bottom_left_corner_percentage_y = 1
bottom_right_corner_percentage_y = 1
theta = np.pi/180     
rho = 1
threshold = 50
min_line_len = 25
max_line_gap = 125
line_color = [0, 255, 0]
line_thickness = 11
weight_param1 = 0.8
weight_param2 = 1.0
weight_param3 = 0.0

# Grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur
def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Canny transform
def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

# Generate a image of mask
def generate_mask(image):
    dimension = image.shape
    top_left = [math.floor(top_left_corner_percentage_x * dimension[1]), math.floor(top_left_corner_percentage_y * dimension[0])]
    top_right = [math.floor(top_right_corner_percentage_x * dimension[1]), math.floor(top_right_corner_percentage_y * dimension[0])]
    bottom_left = [math.floor(bottom_left_corner_percentage_x * dimension[1]), math.floor(bottom_left_corner_percentage_y * dimension[0])]
    bottom_right = [math.floor(bottom_right_corner_percentage_x * dimension[1]), math.floor(bottom_right_corner_percentage_y * dimension[0])]
    shape = np.array([top_left, top_right, bottom_right, bottom_left])
    mask = np.zeros_like(image)
    fill_color = 255
    cv2.fillPoly(mask, np.array([shape]), fill_color)
    return mask

# Apply the mask to the image
def apply_mask(image, mask):
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Hough transform
def hough_transform(image, rho, theta, threshold, min_line_len, max_line_gap):
    hough_lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
    return hough_lines

# Draw lines based on the input lines
def draw_lines(image, lines, color, thickness):
    line_image = np.zeros((*image.shape, 3), dtype=np.uint8)
    left_lane_lines = []
    right_lane_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if (slope < 0):
                left_lane_lines.append(line)
            else:
                right_lane_lines.append(line)
    # Draw the line in maximum length
    left_x1_min = np.amin(np.array(left_lane_lines)[:,:,0])
    left_y1_max = np.amax(np.array(left_lane_lines)[:,:,1])
    left_x2_max = np.amax(np.array(left_lane_lines)[:,:,2])
    left_y2_min = np.amin(np.array(left_lane_lines)[:,:,3])
    right_x1_min = np.amin(np.array(right_lane_lines)[:,:,0])
    right_y1_min = np.amin(np.array(right_lane_lines)[:,:,1])
    right_x2_max = np.amax(np.array(right_lane_lines)[:,:,2])
    right_y2_max = np.amax(np.array(right_lane_lines)[:,:,3])  
    cv2.line(line_image, (left_x1_min, left_y1_max), (left_x2_max, left_y2_min), color, thickness)
    cv2.line(line_image, (right_x1_min, right_y1_min), (right_x2_max, right_y2_max), color, thickness)
    return line_image

# Combine two images in some weight
def weighted_img(lines, image, a, b, c):
    return cv2.addWeighted(image, a, lines, b, c)

# Read a list of images
images = os.listdir(path)
# Choose the target image
image_original = mpimg.imread(path + images[image_index]);
imgplot = plt.imshow(image_original)
plt.show()
# Process the image and plot them
image_gray = grayscale(image_original)
imgplot = plt.imshow(image_gray, cmap = "gray")
plt.show()
image_blur = gaussian_blur(image_gray, kernel_size);
imgplot = plt.imshow(image_blur, cmap = "gray")
plt.show()
image_canny = canny(image_blur, canny_low, canny_high)
imgplot = plt.imshow(image_canny, cmap = "gray")
plt.show()
mask_shape = generate_mask(image_canny)
imgplot = plt.imshow(mask_shape, cmap = "gray")
plt.show()
image_masked =  apply_mask(image_canny, mask_shape)
imgplot = plt.imshow(image_masked, cmap = "gray")
plt.show()
hough_lines = hough_transform(image_masked, rho, theta, threshold, min_line_len, max_line_gap)
lines_image = draw_lines(image_masked, hough_lines, line_color, line_thickness)
imgplot = plt.imshow(lines_image)
plt.show()
final_image = weighted_img(lines_image, image_original, weight_param1, weight_param2, weight_param3)
# Plot the final image
imgplot = plt.imshow(final_image)
plt.show()
