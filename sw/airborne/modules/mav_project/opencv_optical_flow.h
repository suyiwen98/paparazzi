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
 * @file "modules/computer_vision/cv_opencvdemo.h"
 * @author C. De Wagter
 * A simple module showing what you can do with opencv on the bebop.
 */

#ifndef OPENCV_EXAMPLE_H
#define OPENCV_EXAMPLE_H

#ifdef __cplusplus
extern "C" {
#endif

void opencv_optical_flow(char *img, int width, int height, char *img_prev, int width_prev, int height_prev);

//cv::Mat filter_color(cv::Mat image, int y_low, int y_high, int u_low, int u_high, int v_low, int v_high);

#ifdef __cplusplus
}
#endif

#endif

