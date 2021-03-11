#get opencv library
import cv2;
#get matplotlib for plotting graphs
import matplotlib.pyplot as plt;
#get numpy
import numpy as np;
import glob


def filter_color(image_name, y_low, y_high, u_low, u_high, v_low, v_high, resize_factor):
    im = cv2.imread(image_name);
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # src: source, original or input image
    # dsize: desired size for the output image
    # fx: scale factor along the horizontal axis
    im = cv2.rotate(im, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    im = cv2.resize(im, (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor)));
    # convert an image from RBG space to YUV.
    YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV);
    Filtered = np.zeros([YUV.shape[0], YUV.shape[1]]);
    for y in range(YUV.shape[0]):
        for x in range(YUV.shape[1]):
            if (YUV[y, x, 0] >= y_low and YUV[y, x, 0] <= y_high and \
                    YUV[y, x, 1] >= u_low and YUV[y, x, 1] <= u_high and \
                    YUV[y, x, 2] >= v_low and YUV[y, x, 2] <= v_high):
                Filtered[y, x] = 1;

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
    image_dir_path='./AE4317_2019_datasets/cyberzoo_poles/20190121-135009/*.jpg'
    filenames = glob.glob(image_dir_path)
    filenames.sort()
    Filtered_pole = filter_color(filenames[0], y_low=50, y_high=200, 
                                        u_low=0, u_high=120, v_low=160, v_high=220, resize_factor=4);
    
    