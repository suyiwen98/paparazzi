#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:03:12 2021

@author: suyiwen
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import calibration

image_dir_path='./AE4317_2019_datasets/cyberzoo_poles/20190121-135009/*.jpg'
filenames = glob.glob(image_dir_path)
filenames.sort()
img = calibration.undistort(filenames[70])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp  = fast.detect(gray, None);
pts = cv2.KeyPoint_convert(kp)
pts = pts.reshape((pts.shape[0], 1, 2))
    
    
img2 =  cv2.drawKeypoints(img, kp, None, color=(255,0,0))
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(img2)
plt.show()