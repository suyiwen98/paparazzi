# get opencv library
import cv2
# get matplotlib for plotting graphs
import matplotlib.pyplot as plt
# get numpy
import numpy as np
import glob
import calibration


def filter_color(image_name, y_low, y_high, u_low, u_high, v_low, v_high, resize_factor):
    im = calibration.undistort(image_name)
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # src: source, original or input image
    # dsize: desired size for the output image
    # fx: scale factor along the horizontal axis

    im = cv2.resize(im, (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor)))
    # convert an image from RBG space to YUV.
    YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    Filtered = np.zeros([YUV.shape[0], YUV.shape[1]])
    for y in range(YUV.shape[0]):
        for x in range(YUV.shape[1]):
            if (y_low <= YUV[y, x, 0] <= y_high and
                    u_low <= YUV[y, x, 1] <= u_high and
                    v_low <= YUV[y, x, 2] <= v_high):
                Filtered[y, x] = 1

    plt.figure();
    RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB);
    plt.imshow(RGB);
    plt.title('Original image');
    
    plt.figure()
    plt.imshow(Filtered);
    plt.title('Filtered image');
    plt.show()
    return Filtered

def get_YCbCr(img):
    """This gets the average Y, Cr, Cb values of the green grass part from the 
    first image
    Output:
        Y, Cr, Cb: Average Y, Cr, Cb integer values"""
    
    im=calibration.undistort(img)
    # convert an image from RBG space to YCbCr.
    YCrCb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    
    #get average Y value from the green part
    Y=int(np.average(YCrCb[:,:,0]))
    Y_std = int(np.std(YCrCb[:,:,0]))
    #get average Cb value
    Cr=int(np.average(YCrCb[:,:,1]))
    Cr_std = int(np.std(YCrCb[:,:,1]))
    #get average Cr value
    Cb=int(np.average(YCrCb[:,:,2]))
    Cb_std = int(np.std(YCrCb[:,:,2]))
    
    print([Y,Cr,Cb],[2*Y_std, 2*Cr_std, 2*Cb_std])
    return Y, Cr, Cb

if __name__ == '__main__':
    image_dir_path = '/home/agrim/Downloads/dataset/*.jpeg'
    #get_YCbCr(image_dir_path)
    filenames = glob.glob(image_dir_path)
    filenames.sort()
    ''' img = filenames[6]
    Filtered_pole = filter_color(img, y_low=90, y_high=130,u_low=80, u_high=110, v_low=120, v_high=145, resize_factor=4)
    '''
    for img in filenames:
        Filtered_pole = filter_color(img, y_low=60, y_high=140,u_low=75, u_high=110, v_low=120, v_high=145, resize_factor=4)
        input()