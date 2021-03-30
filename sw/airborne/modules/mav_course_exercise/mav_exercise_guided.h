/*
 * Copyright (C) Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/mav_course_exercise/mav_exercise_guided.h"
 * @author Marko Reinhard, Agrim Sharma
 * This module is the contribution of Group 1 for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the guided mode of the autopilot.

 *
 * The color filter settings are set using the cv_detect_color_object. This filter has been modified.
 * One filter is used to detect the boundaries of the Cyberzoo using the bottom camera, and the other filter is used to count the green pixels in three different image segments.
 * If the middle image segment has the highest number of green pixels, the direction is kept. Otherwise, the MAV turns towards the side with the higher green pixel count.
 * Please be aware that the filter file in the computer vision module has been modified to fit the needs. The decision if the drone turns is made there, and published via an ABI message.
 */


#ifndef PAPARAZZI_MAV_EXERCISE_GUIDED_H
#define PAPARAZZI_MAV_EXERCISE_GUIDED_H

#include <stdint.h>

// settings
extern float oag_color_count_frac;  // obstacle detection threshold as a fraction of total of image
extern float oag_floor_count_frac;  // floor detection threshold as a fraction of total of image
extern float oag_max_speed;         // max flight speed [m/s]
extern float oag_heading_rate;      // heading change in case obstacle is detected [rad/s]

extern float oob_heading_rate; 	    // heading change rate in case drone is out of Cyberzoo boundaries [rad/s]

extern void mav_exercise_guided_init(void);
extern void mav_exercise_guided_periodic(void);

#endif

