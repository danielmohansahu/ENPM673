#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 08:58:23 2020

@author: Brenda

working code for proj2
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


image = cv2.imread("Frame 1 copy.jpg")
count = 0
cv2.imshow("original",image )
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray )
cv2.waitKey(0)
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
lower_yellow = np.array([20, 100, 100], dtype = "uint8")
upper_yellow = np.array([30, 255, 255], dtype="uint8")

maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

maskWhite = cv2.inRange(gray, 200, 255)

maskYW = cv2.bitwise_or(maskWhite, maskYellow)

maskImageYW = cv2.bitwise_and(gray, maskYW)

cv2.imshow("HSV",maskImageYW )
cv2.waitKey(0)

kernel = 5
gauss = cv2.GaussianBlur(maskImageYW, (kernel, kernel), 0)

cv2.imshow("Gauss",gauss )
cv2.waitKey(0)

low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gauss, low_threshold, high_threshold)

lower_left = [0 ,720]
print(lower_left)
lower_right = [1280,720]
print(lower_right)
top_left = [0,430]
print(top_left)
top_right = [1280, 430]
print(top_right)
vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]


#ROI
mask = np.zeros_like(canny_edges)   
    
if len(gauss.shape) > 2:
        channel_count = canny_edges.shape[2]  
        ignore_mask_color = (255,) * channel_count
else:
        ignore_mask_color = 255
        
    #filling pixels   
cv2.fillPoly(mask, vertices, ignore_mask_color)
    
masked_image = cv2.bitwise_and(canny_edges, mask)

cv2.imshow("ROI", masked_image )
cv2.waitKey(0)

#####  NOT WORKING YET
print(vertices)
rho = 4
theta = np.pi/180
#threshold is minimum number of intersections 
threshold = 30
min_line_len = 50  #guess???
max_line_gap = 150


lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
line_img = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
for line in lines:

    x1, y1, x2, y2 = line[0]

cv2.line(masked_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
cv2.waitKey(0)