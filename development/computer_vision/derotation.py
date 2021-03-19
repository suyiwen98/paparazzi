#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:25:29 2021

@author: suyiwen
"""

import pandas as pd
import glob
import os 
import re

"""A,B,C rotational velocities of cameral and x,y image coordinates"""
df=pd.read_csv(r'./AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142943.csv')
image_dir_path='./AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/*.jpg'
filenames = glob.glob(image_dir_path)
filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

times=[]
#get timestamps from image names
for image_name in filenames:
    name  = os.path.split(image_name)[-1]
    name  =os.path.splitext(name)[0]
    time  =float(name[0:2]+'.'+name[2:])
    times.append(time)
    
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
print(A)
    
def rotation(A,B,C,x,y):

    #rotational component of horizontal flow
    ur = A*x*y-B*x**2-B+C*y
    #rotational component of horizontal flow
    vr = -C*x+A+A*y**2-B*x*y
    
    return ur,vr
    