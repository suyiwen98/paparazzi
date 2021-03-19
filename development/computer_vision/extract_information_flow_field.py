#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:34:32 2021

Script that can be run on a directory, calculates optical flow and extracts useful information from the flow field.

@author: Guido de Croon, modified by Suyi Wen
"""
#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import re
import os
import pandas as pd
import random as rng

#import calibration script
import calibration

def extract_features(img,gray,boundRect):
    """Detect features on rectangular regions of interest and returns
    coordinates of detected features
    Input: 
        img: callibrated image
        gray: gray scale filtered image
        boundRect: parameters of rectangle of frontal pole (x of top left corner,
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
    
#    #contour of max height
#    pts=front_contours
#    #generte random indices
#    idx = np.random.randint(len(pts), size=7)
#    #get the coordinates of random indices
#    pts = pts[idx,:]
#    pts = pts.reshape(-1,pts.shape[2])
#    for i in range(len(pts)):
#        cv2.circle(img, tuple(pts[i]), radius=5, color=(255,0,0), thickness=-1)
#    plt.figure()
#    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#    plt.title('Detected features')
    #pts = pts.reshape((pts.shape[0], 1, 2))
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create(threshold = 1)
    
    #create a mask of white poles and black background
    mask = [[0]*gray.shape[1]]*gray.shape[0]
    mask = np.asarray(mask)
    mask = mask.astype(np.uint8) 
    (x,y,w,h) = boundRect
    
    if gray.shape[0]>x>0 and gray.shape[1]>y>0:
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

def derotation(A,B,C,points,flow_vectors):
    """Returns the translational optical flow vectors after subtracting the rotational components from total flow
    inputs: 
        A,B,C: rotational rates of the camera
        flow_vectors: total optical flow vectors"""
    for i in range(len(points[0])):
        x=points[0][i]
        y=points[1][i]
        
        #rotational component of horizontal flow
        ur = A*x*y-B*x**2-B+C*y
        #rotational component of vertical flow
        vr = -C*x+A+A*y**2-B*x*y
        
        flow_vectors[0][i]=flow_vectors[0][i]-ur
        flow_vectors[1][i]=flow_vectors[1][i]-vr
        
    return flow_vectors

def filter_color(im, y_low, y_high, u_low, u_high, v_low, v_high):
    """This filters the image based on YUV thresholds and returns binary filter
    inputs:
        im: bgr colored image
        y_low,y_high, u_low, u_high, v_low, v_high: YUV thresholds"""
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

def determine_optical_flow(prev_bgr, bgr,prev_bgr_time,bgr_time, graphics= True):
        
    # convert the images to grayscale and blur it to remove noise
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY);
    prev_gray = cv2.blur(prev_gray, (3,3))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY);
    gray = cv2.blur(gray, (3,3))
    
    # initial threshold
    thresh = 20
        
    # Pole Detector
    Filtered = filter_color(prev_bgr, y_low = 50, y_high = 250, u_low = 0, u_high = 150, 
                                v_low = 140, v_high = 220);
    
    # Reduce noise on detected pole
    Filtered = np.uint8(Filtered)
    reduced_noise = cv2.fastNlMeansDenoisingColored(Filtered, None, 90, 10, 7, 21)
    ret, reduced_noise = cv2.threshold(reduced_noise, 250, 255, cv2.THRESH_BINARY)
    
    # find the contours and rectangular areas of interest
    img_contoured,boundRect,front_contours= get_front_contour(reduced_noise,thresh)
    
    #FAST feature detection
    points_old = extract_features(prev_bgr,prev_gray,boundRect)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
    
    # calculate optical flow
    points_new, status, error_match = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points_old, None, **lk_params)
    
    # filter the points by their status:
    points_old = points_old[status == 1];
    points_new = points_new[status == 1];
    
    flow_vectors = points_new - points_old;
        
    if(graphics):
        im = (0.5 * prev_bgr.copy().astype(float) + 0.5 * bgr.copy().astype(float)) / 255.0;
        im = cv2.cvtColor(np.float32(im), cv2.COLOR_BGR2RGB)
        n_points = len(points_old);
        
        #RGB color of optical flow arrow
        color = (1.0,1.0,1.0);
        #line thickness of 3 px
        thickness = 2
        for p in range(n_points):
            cv2.arrowedLine(im, tuple(points_old[p, :]), tuple(points_new[p,:]), color, thickness);

        plt.figure();
        plt.imshow(im);
        plt.title('Optical flow from '+str(prev_bgr_time) +"s to "+str(bgr_time)+" s");

    return points_old, points_new, flow_vectors;

def estimate_linear_flow_field(points_old, flow_vectors, RANSAC=True, n_iterations=100, error_threshold=10.0):
    """Estimate the linear flow vectors
    inputs:
        points_old: previous coordinates of detected features
        flow_vectors: optical flow vectors
        RANSAC: use RANSAC for more robust calculations
        n_iterations: number of iterations before stopping
        error_threshold: when the error reaches this number, it stops"""
    n_points = points_old.shape[0];
    sample_size = 3; # minimal sample size is 3
    
    if(n_points >= sample_size):
        
        if(not RANSAC):
               
            # estimate a linear flow field for horizontal and vertical flow separately:
            # make a big matrix A with elements [x,y,1]
            A = np.concatenate((points_old, np.ones([points_old.shape[0], 1])), axis=1);
            
            # Moore-Penrose pseudo-inverse:
            # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
            pseudo_inverse_A = np.linalg.pinv(A);
            
            # target = horizontal flow translational component:
            u_vector = flow_vectors[:,0];
            # solve the linear system:
            pu = np.dot(pseudo_inverse_A, u_vector);
            # calculate how good the fit is:
            errs_u = np.abs(np.dot(A, pu) - u_vector);
            
            # target = vertical flow:
            v_vector = flow_vectors[:,1];
            pv = np.dot(pseudo_inverse_A, v_vector);
            errs_v = np.abs(np.dot(A, pv) - v_vector);
            err = (np.mean(errs_u) + np.mean(errs_v)) / 2.0;
            
        else:
            # This is a RANSAC method to better deal with outliers
            # matrices and vectors for the big system:
            A = np.concatenate((points_old, np.ones([points_old.shape[0], 1])), axis=1);
            u_vector = flow_vectors[:,0];
            v_vector = flow_vectors[:,1];
            
            # solve many small systems, calculating the errors:
            errors = np.zeros([n_iterations, 2]);
            pu = np.zeros([n_iterations, 3])
            pv = np.zeros([n_iterations, 3])
            for it in range(n_iterations):
                inds = np.random.choice(range(n_points), size=sample_size, replace=False);
                AA = np.concatenate((points_old[inds,:], np.ones([sample_size, 1])), axis=1);
                pseudo_inverse_AA = np.linalg.pinv(AA);
                # horizontal flow:
                u_vector_small = flow_vectors[inds, 0];
                # pu[it, :] = np.linalg.solve(AA, UU);
                pu[it,:] = np.dot(pseudo_inverse_AA, u_vector_small);
                errs = np.abs(np.dot(A, pu[it,:]) - u_vector);
                errs[errs > error_threshold] = error_threshold;
                errors[it, 0] = np.mean(errs);
                # vertical flow:
                v_vector_small = flow_vectors[inds, 1];
                # pv[it, :] = np.linalg.solve(AA, VV);
                pv[it, :] = np.dot(pseudo_inverse_AA, v_vector_small);
                errs = np.abs(np.dot(A, pv[it,:]) - v_vector);
                errs[errs > error_threshold] = error_threshold;
                errors[it, 1] = np.mean(errs);
            
            # take the minimal error
            errors = np.mean(errors, axis=1);
            ind = np.argmin(errors);
            err = errors[ind];
            pu = pu[ind, :];
            pv = pv[ind, :];
    else:
        # not enough samples to make a linear fit:
        pu = np.asarray([0.0]*3);
        pv = np.asarray([0.0]*3);
        err = error_threshold;
        
    return pu, pv, err;

def get_all_image_names(image_dir_path):
    """
    returns a sorted list of all image names;
    input: <image_dir_name> is the folder path, 
    for example, './AE4317_2019_datasets/calibration_frontcam/20190121-163447/*.jpg'
    """
    image_names =  glob.glob(image_dir_path);
    image_names.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    times=[]
    #get timestamps from image names
    for image_name in image_names:
        image_name  = os.path.split(image_name)[-1]
        image_name  =os.path.splitext(image_name)[0]
        time  =float(image_name[0:2]+'.'+image_name[2:])
        times.append(time)
        
    return image_names,times
        
def extract_flow_information(image_dir_path, df, verbose=True, graphics = True, flow_graphics = False):
    
    image_names,times=get_all_image_names(image_dir_path)   
    
    #extract time stamps fnad rotational rates from csv file
    t          = df["time"]      
    roll_rate  = df["rate_p"]
    yaw_rate   = df["rate_r"]
    pitch_rate = df["rate_q"]
    
    A = []      #pitch rate
    B = []      #-roll rate?
    C = []      #yaw rate  
    
    for j in range(len(times)):
        idx =next(x for x, val in enumerate(t) if val >= times[j]) 
        A.append(roll_rate[idx])
        B.append(yaw_rate[idx])
        C.append(pitch_rate[idx])
    
    # iterate over the images:
    n_images = len(image_names);
    FoE_over_time = np.zeros([n_images, 2]);
    horizontal_motion_over_time = np.zeros([n_images, 1]);
    vertical_motion_over_time = np.zeros([n_images, 1]);
    divergence_over_time = np.zeros([n_images, 1]);
    errors_over_time = np.zeros([n_images, 1]);
    elapsed_times = np.zeros([n_images,1]);
    ttc_over_time = np.zeros([n_images,1]);
    FoE = np.asarray([0.0]*2);

    #starting from the 300th image from the dataset
    start = 310
    end   = 315 #max is n_images
    
    for im in np.arange(start, end, 1):
    
        #calibrates and rotates the image
        bgr = calibration.undistort(image_names[im])
        bgr_time = times[im]
        
        #resize image for faster processing
        resize_factor = 1
        bgr = cv2.resize(bgr, (int(bgr.shape[1] / resize_factor), int(bgr.shape[0] / resize_factor)));
        
        if(im > start):
            
            try: 
                t_before = time.time()
    
                # determine translational optical flow:
                points_old, points_new, flow_vectors = determine_optical_flow(prev_bgr, bgr, prev_bgr_time,bgr_time, graphics=flow_graphics);
                
                #get translational optical flow
                flow_vectors=derotation(A[im],B[im],C[im],points_old,flow_vectors)
                
                # do stuff
                elapsed = time.time() - t_before;
                if(verbose):
                    print('Elapsed time = {}'.format(elapsed));
                elapsed_times[im] = elapsed;
      
                # convert the pixels to a frame where the coordinate in the center is (0,0)
                points_old -= 128.0;
                
                # extract the (3 parameters fit) parameters of the flow field:
                pu, pv, err = estimate_linear_flow_field(points_old, flow_vectors);
                
                # ************************************************************************************
                # extract the focus of expansion and divergence from the flow field:
                # ************************************************************************************
                horizontal_motion = -pu[2];  #u=ax+c
                vertical_motion = -pv[2];    #v=by+c
                divergence = (pu[0]+pv[1]) / 2.0; # 0.0;
                
                small_threshold = 1E-5;
                if(abs(pu[0]) > small_threshold):
                     FoE[0] = pu[2] / pu[0]; 
                if(abs(pv[1]) > small_threshold):
                     FoE[1] = pv[2] / pv[1];    
                if(abs(divergence) > small_threshold):
                     time_to_contact = 1 / divergence;
                        
                # book keeping:
                horizontal_motion_over_time[im] = horizontal_motion;
                vertical_motion_over_time[im] = vertical_motion;
                FoE_over_time[im, 0] = FoE[0];
                FoE_over_time[im, 1] = FoE[1];
                divergence_over_time[im] = divergence;
                errors_over_time[im] = err;
                ttc_over_time[im] = time_to_contact;
                
                if(verbose):
                    # print the FoE and divergence:
                    print('error = {}, FoE = {}, {}, and divergence = {}'.format(err, FoE[0], FoE[1], divergence));
            except:
                plt.figure();
                plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB));
                plt.title('Error at '+str(prev_bgr_time) +"s ");
                
        # the current image becomes the previous image:
        prev_bgr = bgr;
        prev_bgr_time=bgr_time
    
    print('*** average elapsed time = {} ***'.format(np.mean(elapsed_times[1:,0])));
    
    if(graphics):
        plt.figure();
        plt.plot(range(n_images), divergence_over_time, label='Divergence');
        plt.xlabel('Image')
        plt.ylabel('Divergence')

        
        plt.figure();
        plt.plot(range(n_images), FoE_over_time[:,0], label='FoE[0]');
        plt.plot(range(n_images), FoE_over_time[:,1], label='FoE[1]');
        plt.plot(range(n_images), np.asarray([0.0]*n_images), label='Center of the image')
        plt.legend();
        plt.xlabel('Image')
        plt.ylabel('FoE')
        
        plt.figure();
        plt.plot(range(n_images), errors_over_time, label='Error');
        plt.xlabel('Image')
        plt.ylabel('Error')
        
        plt.figure();
        plt.plot(range(n_images), horizontal_motion_over_time, label='Horizontal motion');
        plt.plot(range(n_images), vertical_motion_over_time, label='Vertical motion');
        plt.legend();
        plt.xlabel('Image')
        plt.ylabel('Motion U/Z')    
        
        plt.figure();
        plt.plot(range(n_images), ttc_over_time, label='Time-to-contact');
        plt.xlabel('Image')
        plt.ylabel('Time-to-contact')
       
    return ttc_over_time

if __name__ == '__main__':        
    
    # Change flow_gaphics to True in order to see images and optical flow:
    image_dir_path='./AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/*.jpg'
    #get corresponding csv data
    df=pd.read_csv(r'./AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142943.csv')
    #extract info from optical flow
    extract_flow_information(image_dir_path, df, verbose=True, graphics = True, flow_graphics = True)
