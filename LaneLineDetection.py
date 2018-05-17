# Shuwei Qiang
# May 16, 2018
# Project 1: Simple Lane Line Detection

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip

# Parameters for image processing
image_path = "resource/image/"
video_path = "resource/video/"
kernel_size = 5
canny_low = 75
canny_high = 175
top_left_corner_percentage_x = 0.45
top_right_corner_percentage_x = 0.55
bottom_left_corner_percentage_x = 0.10
bottom_right_corner_percentage_x = 0.90
top_left_corner_percentage_y = 0.6
top_right_corner_percentage_y = 0.6
bottom_left_corner_percentage_y = 1
bottom_right_corner_percentage_y = 1
rho = 1
theta = np.pi/180
threshold = 50
min_line_len = 25
max_line_gap = 125

# Store previous detected lines
left_line_bot_y = 0
left_line_bot_x = 0
left_line_top_y = 0
left_line_top_x = 0
right_line_bot_y = 0
right_line_bot_x = 0
right_line_top_y = 0
right_line_top_x = 0
left_lane_line_slope = 0
right_lane_line_slope = 0

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def generate_quadrilateral_mask(img):
    dimension = img.shape
    top_left = [math.floor(top_left_corner_percentage_x * dimension[1]), math.floor(top_left_corner_percentage_y * dimension[0])]
    top_right = [math.floor(top_right_corner_percentage_x * dimension[1]), math.floor(top_right_corner_percentage_y * dimension[0])]
    bottom_left = [math.floor(bottom_left_corner_percentage_x * dimension[1]), math.floor(bottom_left_corner_percentage_y * dimension[0])]
    bottom_right = [math.floor(bottom_right_corner_percentage_x * dimension[1]), math.floor(bottom_right_corner_percentage_y * dimension[0])]
    mask_shape = np.array([top_left, top_right, bottom_right, bottom_left])
    return np.array([mask_shape])

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def naive_draw_lines(img, lines, color = [255, 0, 0], thickness = 2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines(img, lines, color = [255, 0, 0], thickness = 11):
    global left_line_bot_y
    global left_line_bot_x
    global left_line_top_y
    global left_line_top_x
    global right_line_bot_y
    global right_line_bot_x
    global right_line_top_y
    global right_line_top_x
    global left_lane_line_slope
    global right_lane_line_slope
    left_lane_lines = []
    right_lane_lines = []
    left_slopes = []
    right_slopes = []
    longest_left_line_x = 0;
    longest_left_line_y = 0;
    longest_right_line_x = 0;
    longest_right_line_y = 0;
    longest_left_line_length = 0;
    longest_right_line_length = 0;
    lane_end_precentage_y = top_left_corner_percentage_y + 0.02;
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            length = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if (slope > -0.9 and slope < -0.5):
                left_lane_lines.append(line)
                left_slopes.append(slope)
                if (length > longest_left_line_length):
                    longest_left_line_length = length
                    longest_left_line_x = x1
                    longest_left_line_y = y1
            elif (slope > 0.5 and slope < 0.9):
                right_lane_lines.append(line)
                right_slopes.append(slope)
                if (length > longest_right_line_length):
                    longest_right_line_length = length
                    longest_right_line_x = x1
                    longest_right_line_y = y1
    if (len(left_slopes) != 0):
        if (left_lane_line_slope != 0):
            left_lane_line_slope = (left_lane_line_slope + np.mean(left_slopes)) / 2
        else:
            left_lane_line_slope = np.mean(left_slopes)
        left_line_bot_y = img.shape[0]
        if (left_line_bot_x != 0):
            left_line_bot_x = (left_line_bot_x + longest_left_line_x - ((longest_left_line_y - left_line_bot_y) / left_lane_line_slope)) / 2
        else:
            left_line_bot_x = longest_left_line_x - ((longest_left_line_y - left_line_bot_y) / left_lane_line_slope)
        left_line_top_y = lane_end_precentage_y * img.shape[0]
        left_line_top_x = left_line_bot_x - ((left_line_bot_y - left_line_top_y) / left_lane_line_slope)
    if (len(right_slopes) != 0):
        if(right_lane_line_slope != 0):
            right_lane_line_slope = (right_lane_line_slope + np.mean(right_slopes)) / 2
        else:
            right_lane_line_slope = np.mean(right_slopes)
        right_line_bot_y = img.shape[0]
        if (right_line_bot_x != 0):
            right_line_bot_x = (right_line_bot_x + longest_right_line_x - ((longest_right_line_y - right_line_bot_y) / right_lane_line_slope)) / 2
        else:
            right_line_bot_x = longest_right_line_x - ((longest_right_line_y - right_line_bot_y) / right_lane_line_slope)
        right_line_top_y = lane_end_precentage_y * img.shape[0]
        right_line_top_x = right_line_bot_x - ((right_line_bot_y - right_line_top_y) / right_lane_line_slope)
    cv2.line(img, (math.floor(left_line_top_x), math.floor(left_line_top_y)), (math.floor(left_line_bot_x), math.floor(left_line_bot_y)), color, thickness)
    cv2.line(img, (math.floor(right_line_top_x), math.floor(right_line_top_y)), (math.floor(right_line_bot_x), math.floor(right_line_bot_y)), color, thickness)

def reset_last_lane_line():
    global left_line_bot_y
    global left_line_bot_x
    global left_line_top_y
    global left_line_top_x
    global right_line_bot_y
    global right_line_bot_x
    global right_line_top_y
    global right_line_top_x
    global left_lane_line_slope
    global right_lane_line_slope
    left_line_bot_y = 0
    left_line_bot_x = 0
    left_line_top_y = 0
    left_line_top_x = 0
    right_line_bot_y = 0
    right_line_bot_x = 0
    right_line_top_y = 0
    right_line_top_x = 0
    left_lane_line_slope = 0
    right_lane_line_slope = 0

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, naive = False):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    if (naive):
        naive_draw_lines(line_img, lines)
    else:
        draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a = 0.8, b = 1., c = 0.):
    return cv2.addWeighted(initial_img, a, img, b, c)

def process_image(original_img):
    grayscaled_img = grayscale(original_img)
    blurred_img = gaussian_blur(grayscaled_img, kernel_size)
    edge_img = canny(blurred_img, canny_low, canny_high)
    masked_img = region_of_interest(edge_img, generate_quadrilateral_mask(edge_img))
    detected_lines_image = hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap)
    result = weighted_img(detected_lines_image, original_img)
    return result

def process_vedio(vedio_clip):
    final_clip = vedio_clip.fl_image(process_image)
    return final_clip

# Process a video
# output_name = video_path + "result-solidYellowLeft.mp4"
# clip = VideoFileClip(video_path + "solidYellowLeft.mp4")
# final_clip = process_vedio(clip)
# final_clip.write_videofile(output_name, audio = False)
 
test_image = mpimg.imread(image_path + 'solidwhiteCurve.jpg')
reset_last_lane_line()
plt.imshow(test_image)
plt.show()
grayscaled_img = grayscale(test_image)
plt.imshow(grayscaled_img, cmap='gray')
plt.show()
blurred_img = gaussian_blur(grayscaled_img, kernel_size)
plt.imshow(blurred_img, cmap='gray')
plt.show()
edge_img = canny(blurred_img, canny_low, canny_high)
plt.imshow(edge_img, cmap='gray')
plt.show()
masked_img = region_of_interest(edge_img, generate_quadrilateral_mask(edge_img))
plt.imshow(masked_img, cmap='gray')
plt.show()
detected_naive_lines_image = hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap, True)
plt.imshow(detected_naive_lines_image)
plt.show()
detected_lines_image = hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap)
plt.imshow(detected_lines_image)
plt.show()
final_image = weighted_img(detected_lines_image, test_image)
plt.imshow(final_image)
plt.show()
