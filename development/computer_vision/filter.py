# get opencv library
import cv2
# get matplotlib for plotting graphs
import matplotlib.pyplot as plt
# get numpy
import numpy as np
import glob
import re

import calibration


def filter_color(image_name, y_low, y_high, u_low, u_high, v_low, v_high, resize_factor):
    im = calibration.undistort(image_name)
    #blur the image to remove noise
    im = cv2.blur(im, (10, 10))
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # src: source, original or input image
    # dsize: desired size for the output image
    # fx: scale factor along the horizontal axis

    im = cv2.resize(im, (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor)))
    # convert an image from RBG space to YUV.
    YCrCb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    Filtered = np.zeros([YCrCb.shape[0], YCrCb.shape[1]])
    for y in range(YCrCb.shape[0]):
        for x in range(YCrCb.shape[1]):
            if (y_low <= YCrCb[y, x, 0] <= y_high and
                    u_low <= YCrCb[y, x, 1] <= u_high and
                    v_low <= YCrCb[y, x, 2] <= v_high):
                Filtered[y, x] = 1
    
    ret, Filtered = cv2.threshold(Filtered, 0, 255, cv2.THRESH_BINARY)
    

    plt.figure();
    RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB);
    plt.imshow(RGB);
    plt.title('Original image');
    
    plt.figure()
    plt.imshow(Filtered);
    plt.title('Filtered image');
    plt.show()

    return Filtered


if __name__ == '__main__':
    image_dir_path = './AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/*.jpg'
    filenames = glob.glob(image_dir_path)
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    for i in filenames[450:460]:
        Filtered_pole = filter_color(i, y_low = 50, y_high = 255, u_low = 0, 
                                 u_high = 130, v_low = 0, v_high = 140, resize_factor=1)
