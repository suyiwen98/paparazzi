#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 20:50:11 2021

@author: suyiwen
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import calibration
import random

image_dir_path='./AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/*.jpg'
filenames = glob.glob(image_dir_path)
filenames.sort()
start=300
end=310
threshold = 50 # initial threshold
for im in filenames[start:end]:
    img,name=calibration.undistort(im)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    
    #convert image to gray
    gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Blur the image to reduce noise
    gray =  cv2.blur(gray, (3,3))
    
    #Detect edges using Canny
    canny_output = cv2.Canny(gray, threshold, threshold * 2)
     
    #create an empty image for contours
    _, contours, hierarchy = cv2.findContours(canny_output,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

        
    blank_image = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        
    for i,c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        x,y,w,h = boundRect[i]
        if w>5 and h>20:
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv2.drawContours(blank_image, contours_poly, i, color,thickness=2)
            cv2.rectangle(blank_image, (int(boundRect[i][0]), int(boundRect[i][1])), \
              (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color,2)


    plt.figure()
    plt.imshow(blank_image)
    plt.title("Edges of Image"+str(start))
    
    plt.figure()
    plt.imshow(rgb)
    plt.title("Image"+str(start))
    
    start+=1


