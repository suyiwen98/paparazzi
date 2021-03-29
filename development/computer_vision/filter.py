#import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
#import the calibration python script
import calibration

def get_YCbCr():
    """This gets the average Y, Cr, Cb values of the green grass part from the 
    first image
    Output:
        Y, Cr, Cb: Average Y, Cr, Cb integer values"""
    
    im=calibration.undistort(filenames[0])
    # convert an image from RBG space to YCbCr.
    YCrCb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    
    #get average Y value from the green part
    Y=int(np.average(YCrCb[150:184,:,0]))
    #get average Cb value
    Cr=int(np.average(YCrCb[150:184,:,1]))
    #get average Cr value
    Cb=int(np.average(YCrCb[150:184,:,2]))
    
    print(Y,Cr,Cb)
    return Y, Cr, Cb
    
def filter_color(image_name, y_low, y_high, cr_low, cr_high, cb_low, cb_high, resize_factor):
    """This creates a YCrCb based image filter and returns a binary image that shows
    the pexels within the threshold in white and the other ones in black
    Input: 
        image_name: image path
        y_low:   the lowest luma componenet
        y_high:  the highest luma component
        cb_low:  the lowest blue-difference component
        cb_high: the highest blue-difference component
        cr_high: the lowest red-difference component
        cr_high" the highest red-difference component
        resize_factor: the factor by which the image size is changed (2: image 
                                                                      is halved
    Output:
        Filtered: filtered binary image (0 is black, 255 is white, the x,y coordinate
        of each pixel equals to the array index)"""
    original = calibration.undistort(image_name)
    #blur the image to remove noise
    im = cv2.blur(original, (10, 10))
    
    #resize the image using resize_factor
    # cv2.resize(input image, desired size for the output image)
    im = cv2.resize(im, (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor)))
    
    # convert an image from RBG space to YCbCr.
    YCrCb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    #initialize an array to store the filtered image
    Filtered = np.zeros([YCrCb.shape[0], YCrCb.shape[1]])
    #check which pixel is within the set thresholds
    for y in range(YCrCb.shape[0]):
        for x in range(YCrCb.shape[1]):
            if (y_low <= YCrCb[y, x, 0] <= y_high and
                    cr_low <= YCrCb[y, x, 1] <= cr_high and
                    cb_low <= YCrCb[y, x, 2] <= cb_high):
                Filtered[y, x] = 1
    
    #make the filtered image binary
    ret, Filtered = cv2.threshold(Filtered, 0, 255, cv2.THRESH_BINARY)
    
    #to compare original image and filtered image
    plt.figure();
    RGB = cv2.cvtColor(original, cv2.COLOR_BGR2RGB);
    plt.imshow(RGB);
    plt.title('Original image');
    
    plt.figure()
    plt.imshow(Filtered);
    plt.title('Filtered image with '+values);
    plt.show()

    return Filtered

def grid_3(img):
    y = img.shape[0]
    x = img.shape[1]
    # cv2.line(image, start_point, end_point, color, thickness )
#    img = cv2.line(img, (0, y // 3), (x, y // 3), (255, 255, 255), 1, 1)
#    img = cv2.line(img, (0, 2 * y // 3), (x, 2 * y // 3), (255, 255, 255), 1, 1)
    img = cv2.line(img, (x // 3, 0), (x // 3, y), (255, 255, 255), 1, 1)
    img = cv2.line(img, (2 * x // 3, 0), (2 * x // 3, y), (255, 255, 255), 1, 1)
    return img

if __name__ == '__main__':
    #specify image folder
    image_dir_path = './Test_samples/real_test/*.jpg'
    #get all image paths and sort them
    filenames = glob.glob(image_dir_path)
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    #get the Y, Cb, and Cr values of green pixels
    Y, Cr, Cb = get_YCbCr()
    #define the range of Y, Cr, Cb values can go up or down
    Y_range  = 80
    Cr_range = 5
    Cb_range = 20
    #check if the filter works on the randomly images
    for i in range(len(filenames)): 
        #original values
        values="original values"
        Filtered = filter_color(filenames[i], y_low = 50, y_high = 255, cr_low = 0, 
                             cr_high = 130, cb_low = 0, cb_high = 135, resize_factor=1)
        #my values
        values="improved values"
        Filtered = filter_color(filenames[i], y_low = Y-Y_range, y_high = Y+Y_range, cr_low = 0, 
                             cr_high = Cr+Cr_range, cb_low = 0, cb_high = Cb+Cb_range, resize_factor=1)
        #agrim's values
        values="Agrim's values"
        Filtered = filter_color(filenames[i], y_low = 90, y_high = 140, cr_low = 80, 
                             cr_high = 110, cb_low = 120, cb_high = Cb+Cb_range, resize_factor=145)
    
    print(Y-Y_range, Y+Y_range,Cr+Cr_range,Cb+Cb_range)
