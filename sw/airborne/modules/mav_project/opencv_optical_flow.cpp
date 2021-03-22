/*
 * Copyright (C) C. De Wagter
 *
 * This file is part of paparazzi
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */
/**
 * @file "modules/computer_vision/opencv_example.cpp"
 * @author C. De Wagter
 * A simple module showing what you can do with opencv on the bebop.
 */


#include "opencv_optical_flow.h"


using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>

using namespace cv;

#include "modules/computer_vision/opencv_image_functions.h"

#define PRINT(string, ...) fprintf(stderr, "[opencv_optical_flow->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)

Mat filter_color(Mat image, int y_low, int y_high, int u_low, int u_high, int v_low, int v_high) {
    Mat yuvImage;
    cvtColor(image, yuvImage, CV_BGR2YUV);

    int nRows = yuvImage.rows;
    int nCols = yuvImage.cols;

    for(int r = 0; r < nRows; ++r) {
        for(int c = 0; c < nCols; ++c) {
            // Extract pixel color from image
            Vec3b &yuv = yuvImage.at<Vec3b>(r, c);

            if (y_low <= yuv[0] && yuv[0] <= y_high &&
                u_low <= yuv[1] && yuv[1] <= u_high &&
                v_low <= yuv[2] && yuv[2] <= v_high) {
                yuv[0] = 255;
                yuv[1] = 255;
                yuv[2] = 255;
            } else {
                yuv[0] = 255;
                yuv[1] = 255;
                yuv[2] = 255;
            }
        }
    }
    return yuvImage;
}

char* opencv_optical_flow(char *img, int width, int height, char *img_prev, int width_prev, int height_prev) {
    // Create a new image, using the original bebop image.
    Mat M(height, width, CV_8UC3, img);
    Mat image;
    cvtColor(M, image, CV_BGR2GRAY);
    blur(image, image, Size(3, 3));

    Mat M_prev(height_prev, width_prev, CV_8UC3, img_prev);
    Mat image_prev;
    cvtColor(M_prev, image_prev, CV_BGR2GRAY);
    blur(image_prev, image_prev, Size(3, 3));

    int y_low = 50;
    int y_high = 250;
    int u_low = 0;
    int u_high = 150;
    int v_low = 140;
    int v_high = 220;

    Mat filtered = filter_color(M_prev, y_low, y_high, u_low, u_high, v_low, v_high);

    Mat reduced_noise;
    fastNlMeansDenoisingColored(filtered, reduced_noise, 90, 10, 7, 21);
    threshold(reduced_noise, reduced_noise, 250, 255, THRESH_BINARY);

    img = (char *)reduced_noise.data; // Try to write the changed image back to the input, but this doesn't actually work.

    return (char *) reduced_noise.data;
}
