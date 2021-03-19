#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:03:12 2021

@author: suyiwen
"""

import cv2
import glob
import matplotlib.pyplot as plt
import calibration
import numpy as np
import re
import random as rng

# https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html
def thresh_callback(gray,val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(gray, threshold, threshold * 2)
    
    # Find contours
    _,contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    boundRect = [None]*len(contours)
    
    for i in range(len(contours)):
        # approximate the contour
        peri = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.01 * peri, True)
        boundRect[i] = cv2.boundingRect(approx)  #(x,y,w,h)
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        
    # Change contour colors to Blue
    drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    
    # Make the Image Binary
    ret, drawing = cv2.threshold(drawing, 1, 255, cv2.THRESH_BINARY)
    # Show in a window
    kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(drawing, kernel, iterations=1)
    closing = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, kernel)
    Contoured = closing
    return Contoured, boundRect

def filter_color(im, y_low, y_high, u_low, u_high, v_low, v_high):
    """This filters the image based on YUV thresholds"""

    # convert an image from RBG space to YUV.
    YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV);
    Filtered = np.zeros([YUV.shape[0], YUV.shape[1], 3]);
    for y in range(YUV.shape[0]):
        for x in range(YUV.shape[1]):
            if (YUV[y, x, 0] >= y_low and YUV[y, x, 0] <= y_high and \
                    YUV[y, x, 1] >= u_low and YUV[y, x, 1] <= u_high and \
                    YUV[y, x, 2] >= v_low and YUV[y, x, 2] <= v_high):
                Filtered[y, x, 0] = 1;
                Filtered[y, x, 1] = 1;
                Filtered[y, x,2] = 1;
                
    # Make the Image Binary
    ret, Filtered = cv2.threshold(Filtered, 0, 255, cv2.THRESH_BINARY)

    return Filtered

def extract_features(img, reduced_noise):
    """Detect features on rectangular regions of interest and returns
    coordinates of detected features
    Input: 
        img: callibrated image
        gray: gray scale filtered image
        boundRect: parameters of a rectangle (x of top left corner,
        y of the same corner, width, height)"""
    white = [255, 255, 255]
    y,x=np.where(np.all(reduced_noise==white,axis=2))
#    x_min=min(x)
#    x_max=max(x)
#    y_min=min(y)
#    y_max=max(y)

    pts = np.column_stack((x,y))
    mask = [[0]*reduced_noise.shape[1]]*reduced_noise.shape[0]
    mask = np.asarray(mask)
    mask = mask.astype(np.uint8) 

    for i in range(len(pts)):
        mask[pts[i][1],pts[i][0]]=255
    plt.figure()
    plt.imshow(mask)
    plt.title("mask")
    
    #random index
    idx = np.random.randint(len(pts), size=10)
    pts = pts[idx,:]

    for i in range(len(pts)):
        print(tuple(pts[i]))
        cv2.circle(img, tuple(pts[i]), radius=5, color=(255,0,0), thickness=-1)
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Detected features of image' +str(start))
    pts = pts.reshape((pts.shape[0], 1, 2))
    return pts

image_dir_path='./AE4317_2019_datasets/cyberzoo_poles/20190121-135009/*.jpg'
filenames = glob.glob(image_dir_path)
filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

start=70
end=85

for im in filenames[start:end]:
    img=calibration.undistort(im)
    resize_factor = 1
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # src: source, original or input image
    # dsize: desired size for the output image
    # fx: scale factor along the horizontal axis
    img = cv2.resize(img, (int(img.shape[1] / resize_factor), int(img.shape[0] / resize_factor)));
    
    origin = img.copy()
    
    #convert image to gray and blur it
    gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3,3))
            
    thresh = 28  # initial threshold
    # cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
    img_contoured,boundRect = thresh_callback(gray,thresh)

    # Pole Detector
    Filtered = filter_color(img, y_low = 50, y_high = 200, u_low = 0, u_high = 120, 
                            v_low = 160, v_high = 220);

    # Reduce noise on detected pole
    Filtered = np.uint8(Filtered)
    reduced_noise = cv2.fastNlMeansDenoisingColored(Filtered, None, 90, 10, 7, 21)
    ret, reduced_noise = cv2.threshold(reduced_noise, 250, 255, cv2.THRESH_BINARY)

    # find the contours    
    img_contoured,boundRect= thresh_callback(reduced_noise,thresh)
    
    #detect features on poles
    pts=extract_features(img, reduced_noise)
    
            
    # Show Results
#    plt.figure()
#    plt.imshow(img)
#    plt.title('1) original'+str(start))
    
    plt.figure()
    plt.imshow(Filtered)
    plt.title('pole detector'+str(start))
    
    plt.figure()
    plt.imshow(reduced_noise)
    plt.title('Noise Reduction' +str(start))
                
    plt.figure()
    plt.imshow( img_contoured)
    plt.title('Contour detector' +str(start))
    
    
    start+=1
