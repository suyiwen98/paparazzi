#import libraries
import cv2;
import numpy as np;
import glob
import random as rng
import matplotlib.pyplot as plt
import re
#import calibration library
import calibration


def grid_3x3(img):
    y = img.shape[0]
    x = img.shape[1]
    # cv2.line(image, start_point, end_point, color, thickness )
    img = cv2.line(img, (0, y//3), (x, y//3), (255, 255, 255), 1, 1)
    img = cv2.line(img, (0, 2*y//3), (x, 2*y//3), (255, 255, 255), 1, 1)
    img = cv2.line(img, (x//3, 0), (x//3, y), (255, 255, 255), 1, 1)
    img = cv2.line(img, (2*x//3, 0), (2*x//3, y), (255, 255, 255), 1, 1)
    return img


# https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html
def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(cnt_gray, threshold, threshold * 2)
    # Find contours
    _,contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Change contour colors to Blue
    drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    # Make the Image Binary
    ret, drawing = cv2.threshold(drawing, 1, 255, cv2.THRESH_BINARY)
    # Show in a window
    kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(drawing, kernel, iterations=1)
    closing = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, kernel)
    Contoured = closing
    return Contoured


def filter_color(image_name, y_low, y_high, u_low, u_high, v_low, v_high, resize_factor):
    im = calibration.undistort(image_name);
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # src: source, original or input image
    # dsize: desired size for the output image
    # fx: scale factor along the horizontal axis

    im = cv2.resize(im, (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor)));
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
    # plt.figure();
    # RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB);
    # plt.imshow(RGB);
    # plt.title('Original image');
    #
    # plt.figure()
    # plt.imshow(Filtered);
    # plt.title('Filtered image');
    # plt.show()
    return Filtered


if __name__ == '__main__':
    # Load images
    image_dir_path='./AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/*.jpg'
    #image_dir_path = './custom_set/*.jpg'
    filenames = glob.glob(image_dir_path)
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    initialization_flag = 0
    # Read Images
    for img_file in filenames:
        img = calibration.undistort(img_file);
        resize_factor = 1
        img = cv2.resize(img, (int(img.shape[1] / resize_factor), int(img.shape[0] / resize_factor)));

        # Find Contours
        cnt_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cnt_gray = cv2.blur(cnt_gray, (3, 3))
        # source_window = 'Source'
        # cv2.namedWindow(source_window)
        # cv2.imshow(source_window, cnt_gray)
        # max_thresh = 255
        thresh = 28  # initial threshold
        # cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
        img_contoured = thresh_callback(thresh)

        # Pole Detector
        Filtered_pole = filter_color(img_file, y_low=70, y_high=100,
                                        u_low=110, u_high=130, v_low=110, v_high=140, resize_factor=1);

        # Reduce noise on detected grass
        Filtered_pole = np.uint8(Filtered_pole)
        reduced_noise = cv2.fastNlMeansDenoisingColored(Filtered_pole, None, 90, 10, 7, 21)
        ret, reduced_noise = cv2.threshold(reduced_noise, 250, 255, cv2.THRESH_BINARY)

        # find the contours
        cnt_gray = reduced_noise
        img_contoured = thresh_callback(thresh)
        # apply grid
        img_contoured = grid_3x3(img_contoured)

        # Show Results
        cv2.imshow('1) source', img)
        cv2.imshow('2) grass detector', Filtered_pole)
        cv2.imshow('3) Noise Reduction', reduced_noise)
        cv2.imshow('4) contour detector', img_contoured)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # close on ESC key
            cv2.destroyAllWindows()