#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 16:20:15 2021

source: https://opencv-python-tutroals.readthedocs.io/en/
latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

@author: Ioannis & Suyi
"""
#Import the required libraries
import numpy as np
import cv2
import glob


# Setup for Calibration
# termination criteria
# *****************************************************************
# TODO: check OpenCV documentation for the termination parameters
# *****************************************************************
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(image_dir_path="AE4317_2019_datasets/calibration_frontcam/20190121-163447/*.jpg", square_size=0.03515, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path.
    parameters: 
        image_dir_name: directory where we get the images. Default is 
        "AE4317_2019_datasets/calibration_frontcam/20190121-163447/*.jpg";
        square_size   : size of a square of a chess board in meters;
        width         : Number of intersection points of squares in the long side 
        of the chess board. Default is 9. 
        height: Number of intersection points of squares in the short side of the 
        chess board. Default is 6. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    
    objp = objp * square_size 
    
    # Arrays to store object points and image points from all the images.
    objpoints = []  # matrix that holds chessboard corners in the 3D world
    imgpoints = []  # 2d points in image plane.
    
    # Load images from dataset and Rotate Them
    filenames = glob.glob(image_dir_path)
    filenames.sort()
    images = [cv2.imread(img) for img in filenames]
    
    #store rotated images
    goodImg = []
    
    for img in images:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width,height), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            goodImg.append(img)
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (width,height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            # will show the image in a window 
    #     cv2.imshow('image', img) 
    #     k = cv2.waitKey(0) & 0xFF
    #     # wait for ESC key to exit 
    #     if k == 27:  
    #         cv2.destroyAllWindows() 
    cv2.destroyAllWindows()
    #Calculate the camera matrix, distortion coefficients, rotation and translation vectors etc
    #it looks for the number of corners and if writter wrongly it can't find the chessboard
    #check the ret value for that
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    #calculate the reprojection error
    #Re-projection error gives a good estimation of just how exact the found parameters are. 
    #The closer the re-projection error is to zero, the more accurate the parameters we found are. 
    tot_error = 0
    for i in range(len(objpoints)):
        
        #transform the object point to image point
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        
        #calculate the absolute norm between what we got with our transformation and 
        #the corner finding algorithm
        error      = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error  += error
        
    #average error is the arithmetical mean of the errors calculated for all the calibration images.
    mean_error = tot_error/len(objpoints)
    print("mean error: ", mean_error)
    
    return [ret, mtx, dist, rvecs, tvecs]
    
def undistort(image_name, mtx, dist):
    """
    It takes an image and undistort it. 
        image_name: full image path name;
        mtx       : camera matrix;
        dist      : distortion coefficients; 
    """
    
    img = cv2.imread(image_name)
    h,  w = img.shape[:2]
    
    #refine the camera matrix based on a free scaling parameter
    #alpha=0, it returns undistorted image with minimum unwanted pixels. 
    #alpha=1, all pixels are retained with some extra black images. 
    #This function also returns an image ROI which can be used to crop the result.
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst1 = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)
    
    # Check the Results
    cv2.imshow('Distroted', img)
    cv2.imshow('Undistorted', dst)
    cv2.imshow('Undistroted & Cut', dst1)
    
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        
    return

if __name__ == '__main__':
    
    ret, mtx, dist, rvecs, tvecs = calibrate()
    image_dir_path="AE4317_2019_datasets/calibration_frontcam/20190121-163447/*.jpg"
    filenames = glob.glob(image_dir_path)
    filenames.sort()
    undistort(filenames[40], mtx, dist)