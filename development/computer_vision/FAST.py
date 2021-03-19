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
def get_front_contour(gray,val):
    """Inputs:
        gray: grayscale image
        val: threshold value
    Outputs:
        Contoured: image with all contours
        boundRect: bounding rectangular parameters (x of top left corner, y, width, height)
        c: coordinates of contour with maximum height"""
        
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(gray, threshold, threshold * 2)
    
    # Find contours
    _,contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    boundRect = [None]*len(contours)
    h= [None]*len(contours)
    for i in range(len(contours)):
        # approximate the contour
        peri = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.01 * peri, True)
        boundRect[i] = cv2.boundingRect(approx)  #(x,y,w,h)
        h[i]    =boundRect[i][3]
        #random color for each contour
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    
    #get contours of maximum height
    front_contours=contours[np.argmax(h)]
    #bounded rectable of the contour with max heigt
    boundRect= cv2.boundingRect(front_contours)
    # Change contour colors to Blue
    drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    
    # Make the Image Binary
    ret, drawing = cv2.threshold(drawing, 1, 255, cv2.THRESH_BINARY)
    # Show in a window
    kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(drawing, kernel, iterations=1)
    closing = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, kernel)
    Contoured = closing
    return Contoured, boundRect, front_contours

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

def extract_features(img,gray,boundRect):
    """Detect features on rectangular regions of interest and returns
    coordinates of detected features
    Input: 
        img: callibrated image
        gray: gray scale filtered image
        boundRect: parameters of a rectangle (x of top left corner,
        y of the same corner, width, height)"""
#    white = [255, 255, 255]
#    y,x=np.where(np.all(img_filtered==white,axis=2))
#    #y,x=np.where(img_filtered==255)
#
#    pts = np.column_stack((x,y))
#    print(pts)
#    mask = [[0]*img_filtered.shape[1]]*img_filtered.shape[0]
#    mask = np.asarray(mask)
#    mask = mask.astype(np.uint8) 
#
#    for i in range(len(pts)):
#        mask[pts[i][1],pts[i][0]]=255
#    plt.figure()
#    plt.imshow(mask)
#    plt.title("mask")
    
    #for all contours
#    for i in range(len(contours)):
#        pts=contours[i]
#        #generte random indices
#        idx = np.random.randint(len(pts), size=7)
#        #get the coordinates of random indices
#        pts = pts[idx,:]
#        pts = pts.reshape(-1,pts.shape[2])
#        for i in range(len(pts)):
#            cv2.circle(img, tuple(pts[i]), radius=5, color=(255,0,0), thickness=-1)
    
    fast = cv2.FastFeatureDetector_create(threshold = 1)
    
    #create a mask of white poles and black background
    mask = [[0]*gray.shape[1]]*gray.shape[0]
    mask = np.asarray(mask)
    mask = mask.astype(np.uint8) 
    (x,y,w,h) = boundRect
    
    if x>0 and y>0:
        # Set the selected region within the mask to white (add 10 pixels to height and width)
        mask[y-10:y+h+10, x-10:x+w+10] = 255
    else:
        mask[y:y+h+10, x-10:x+w+10] = 255

    plt.figure()
    plt.imshow(mask)
    plt.title('Mask')
    
    # find the keypoints
    kp  = fast.detect(gray, mask = mask);
    
    #convert the keypoints to array
    pts = cv2.KeyPoint_convert(kp)
    pts = pts.reshape((pts.shape[0], 1, 2))
    
    #draw the keypoints on original callibrated image
    img_pt = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
    plt.figure()
    plt.imshow(cv2.cvtColor(img_pt, cv2.COLOR_BGR2RGB))
    plt.title('Detected features of image' )
    
    return pts


if __name__ == '__main__':
    image_dir_path='./AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/*.jpg'
    filenames = glob.glob(image_dir_path)
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    start=300
    end=305
    
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
    
        # Pole Detector
        Filtered = filter_color(img, y_low = 50, y_high = 250, u_low = 0, u_high = 150, 
                                v_low = 150, v_high = 220);
        
        # Reduce noise on detected pole
        Filtered = np.uint8(Filtered)
        reduced_noise = cv2.fastNlMeansDenoisingColored(Filtered, None, 90, 10, 7, 21)
        ret, reduced_noise = cv2.threshold(reduced_noise, 250, 255, cv2.THRESH_BINARY)
        
        # find the contours and rectangular areas of interest
        img_contoured,boundRect,front_contours= get_front_contour(reduced_noise,thresh)
        
        #FAST feature detection
        points_old = extract_features(img,gray,boundRect)
        
                
        # Show Results
    #    plt.figure()
    #    plt.imshow(img)
    #    plt.title('1) original'+str(start))
        
        plt.figure()
        plt.imshow(Filtered)
        plt.title('pole detector image nr '+str(start))
        
        plt.figure()
        plt.imshow(reduced_noise)
        plt.title('Noise Reduction image nr ' +str(start))
                    
        plt.figure()
        plt.imshow(img_contoured)
        plt.title('Contour detector image nr ' +str(start))
        
        
        start+=1
