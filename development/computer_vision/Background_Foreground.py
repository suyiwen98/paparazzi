#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:43:32 2021

@author: suyiwen
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import calibration

image_dir_path='./AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/*.jpg'
filenames = glob.glob(image_dir_path)
filenames.sort()
bgs_mog = cv2.createBackgroundSubtractorMOG2()
start=70
end=80
for im in filenames[start:end]:
    img=calibration.undistort(im)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #convert image to gray and blur it
    gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray =  cv2.blur(gray, (3,3))
    
    blank_image = np.zeros(img.shape)
    
    # Set the minimum area for a contour
    min_area = 100
    
    #get foreground mask
    fgmask = bgs_mog.apply(img)
    _, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        print(area)
        if 80000>area>min_area:
            cv2.drawContours(blank_image, contours, -1, (0,255,0), thickness=cv2.FILLED)

    plt.figure()
    plt.imshow(blank_image)
    plt.title("Foreground of Image"+str(start))
    
    plt.figure()
    plt.imshow(rgb)
    plt.title("Image"+str(start))
    
    
    start+=1

    
