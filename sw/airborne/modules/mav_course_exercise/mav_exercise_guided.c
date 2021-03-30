/*
 * Copyright (C) Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/mav_course_exercise/mav_exercise_guided.c"
 * @author Marko Reinhard, Agrim Sharma
 * This module is the contribution of Group 1 for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the guided mode of the autopilot.

 *
 * The color filter settings are set using the cv_detect_color_object. This filter has been modified.
 * One filter is used to detect the boundaries of the Cyberzoo using the bottom camera, and the other filter is used to count the green pixels in three different image segments.
 * If the middle image segment has the highest number of green pixels, the direction is kept. Otherwise, the MAV turns towards the side with the higher green pixel count.
 * Please be aware that the filter file in the computer vision module has been modified to fit the needs. The decision if the drone turns is made there, and published via an ABI message.
 */

#include "mav_exercise_guided.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "generated/airframe.h"
#include "state.h"
#include "subsystems/abi.h"
#include <stdio.h>
#include <time.h>

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[mav_exercise_guided->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

void chooseRandomIncrementAvoidance(void);
void turnTowardsGrass(void);

enum navigation_state_t
    {
    SAFE,
    OBSTACLE_FOUND,
    SEARCH_FOR_SAFE_HEADING,
    OUT_OF_BOUNDS,
    REENTER_ARENA
    };

// define settings
float oag_color_count_frac = 0.1f; // obstacle detection threshold as a fraction of total of image
float oag_floor_count_frac = 0.05f; // floor detection threshold as a fraction of total of image
float oag_max_speed = 0.5f;               // max flight speed [m/s]
float oag_heading_rate = RadOfDeg(10.f); // heading change in case obstacle is detected [rad/s]
float avoidance_heading_direction = 1;  // heading change direction for avoidance [-], initialize with 1
float oob_heading_direction = 1;  // heading change direction for out of boundaries [-], initialize with 1
float oob_heading_rate = RadOfDeg(10.f); // heading change rate in case drone is out of Cyberzoo boundaries [rad/s]


// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING; // current state in state machine
int32_t color_count = 0; // orange color count from color filter for obstacle detection
int32_t floor_count = 0; // green color count from color filter for floor detection
int32_t floor_centroid = 0; // floor detector centroid in y direction (along the horizon)
int16_t obstacle_free_confidence = 0; // a measure of how certain we are that the way ahead if safe.

const int16_t max_trajectory_confidence = 5; // number of consecutive negative object detections to be sure we are obstacle free
int16_t turn_dir = 0; // stores turn direction from grass detector filter output


// First filter in cv_detect_color_object.c is used for grass detection for heading, second filter is used for grass detection to stay in boundaries

// This call back will be used to receive the turn command from the green pixel counter
#ifndef GRASS_DETECTOR_VISUAL_DETECTION_ID
#error This module requires two color filters, as such you have to define GRASS_DETECTOR_VISUAL_DETECTION_ID to the orange filter
#error Please define GRASS_DETECTOR_VISUAL_DETECTION_ID to be COLOR_OBJECT_DETECTION1_ID or COLOR_OBJECT_DETECTION2_ID in your airframe
#endif

static abi_event color_detection_ev;
static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
	int16_t __attribute__((unused)) pixel_x,
	int16_t __attribute__((unused)) pixel_y,
	int16_t __attribute__((unused)) pixel_width,
	int16_t __attribute__((unused)) pixel_height,
	int32_t quality,
	int16_t extra)
    {
    color_count = quality;
    turn_dir = extra; // Read turn direction from CV filter (left, middle, right)
    }

#ifndef FLOOR_VISUAL_DETECTION_ID
#error This module requires two color filters, as such you have to define FLOOR_VISUAL_DETECTION_ID to the orange filter
#error Please define FLOOR_VISUAL_DETECTION_ID to be COLOR_OBJECT_DETECTION1_ID or COLOR_OBJECT_DETECTION2_ID in your airframe
#endif
static abi_event floor_detection_ev;
static void floor_detection_cb(uint8_t __attribute__((unused)) sender_id,
	int16_t __attribute__((unused)) pixel_x,
	int16_t pixel_y,
	int16_t __attribute__((unused)) pixel_width,
	int16_t __attribute__((unused)) pixel_height,
	int32_t quality,
	int16_t __attribute__((unused)) extra)
    {
    floor_count = quality;
    floor_centroid = pixel_y;
    }

/*
 * Initialisation function
 */
void mav_exercise_guided_init(void)
    {
    // Initialise random values
    srand(time(NULL));
    chooseRandomIncrementAvoidance();

    // bind our colorfilter callbacks to receive the color filter outputs
    AbiBindMsgVISUAL_DETECTION(GRASS_DETECTOR_VISUAL_DETECTION_ID,
	    &color_detection_ev, color_detection_cb);
    AbiBindMsgVISUAL_DETECTION(FLOOR_VISUAL_DETECTION_ID, &floor_detection_ev,
	    floor_detection_cb);
    }

/*
 * Function that checks it is safe to move forwards, and then sets a forward velocity setpoint or changes the heading
 */
void mav_exercise_guided_periodic(void)
    {
    // Only run the module if we are in the correct flight mode
    if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED)
	{
	navigation_state = SEARCH_FOR_SAFE_HEADING;
	obstacle_free_confidence = 0;
	return;
	}

    // compute current color thresholds
    int32_t color_count_threshold = oag_color_count_frac * front_camera.output_size.w * front_camera.output_size.h;
    int32_t floor_count_threshold = oag_floor_count_frac * front_camera.output_size.w * front_camera.output_size.h;
    float floor_centroid_frac = floor_centroid
	    / (float) front_camera.output_size.h / 2.f;

    VERBOSE_PRINT("Grass detector: Grass pixels: %d", color_count_threshold);
    VERBOSE_PRINT("Grass detector: Direction %d, Confidence: %d \n", turn_dir, obstacle_free_confidence);
    VERBOSE_PRINT("Floor detector: Count %d, Threshold %d, Centroid %f\n", floor_count, floor_count_threshold, floor_centroid_frac);
    VERBOSE_PRINT("Navigation state: %d \n", navigation_state);


    // update our safe confidence using the turn direction (if highest grass count still in center of image, turn direction would be 0)
    if (turn_dir == 0)
    {
       obstacle_free_confidence++;
    }
    else
    {
       obstacle_free_confidence -= 2;  // might need adjustment
    }

    // bound obstacle_free_confidence
    Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

    // either use maximum speed or speed depending on obstacle free confidence
    float speed_sp = fminf(oag_max_speed, 0.2f * obstacle_free_confidence);

    switch (navigation_state)
	{
    case SAFE:
	if (floor_count < floor_count_threshold	|| fabsf(floor_centroid_frac) > 0.12)
	    {
	    navigation_state = OUT_OF_BOUNDS;
	    }
	else if (obstacle_free_confidence == 0)
	    {
	    navigation_state = OBSTACLE_FOUND;
	    }
	else
	    {
	    guidance_h_set_guided_body_vel(speed_sp, 0);
	    }

	break;
    case OBSTACLE_FOUND:
	// stop
	guidance_h_set_guided_body_vel(0, 0);

	// turn in direction of highest grass count
	turnTowardsGrass();

	navigation_state = SEARCH_FOR_SAFE_HEADING;

	break;
    case SEARCH_FOR_SAFE_HEADING:
	guidance_h_set_guided_heading_rate(avoidance_heading_direction * oag_heading_rate);

	// make sure we have a couple of good readings before declaring the way safe
	if (obstacle_free_confidence >= 2)
	    {
	    guidance_h_set_guided_heading(stateGetNedToBodyEulers_f()->psi);
	    navigation_state = SAFE;
	    }
	break;
    case OUT_OF_BOUNDS:
	// stop
	guidance_h_set_guided_body_vel(0, 0);

	// Get random turn direction
	chooseRandomIncrementAvoidance();

	// start turn back into arena
	guidance_h_set_guided_heading_rate(oob_heading_direction * oob_heading_rate);

	navigation_state = REENTER_ARENA;

	break;
    case REENTER_ARENA:
	// force floor center to opposite side of turn to head back into arena
	if (floor_count >= floor_count_threshold && oob_heading_direction * floor_centroid_frac >= 0.f)
	    {
	    // return to heading mode
	    guidance_h_set_guided_heading(stateGetNedToBodyEulers_f()->psi);

	    // reset safe counter
	    obstacle_free_confidence = 0;

	    // ensure direction is safe before continuing
	    navigation_state = SAFE;
	    }
	break;
    default:
	break;
	}
    return;
    }

/*
 * Sets the variable 'incrementForAvoidance' randomly positive/negative
 */
void chooseRandomIncrementAvoidance(void)
    {
    // Randomly choose CW or CCW avoiding direction
    if (rand() % 2 == 0)
	{
	oob_heading_direction = 1.f;
	}
    else
	{
	oob_heading_direction = -1.f;
	}
    return false;
    }

// flips heading direction according to grass detector filter
void turnTowardsGrass(void)
    {
    if (turn_dir < 0)
	{
	if (avoidance_heading_direction > 0)
	    {
	    avoidance_heading_direction = avoidance_heading_direction * -1.f; // flip direction
	    }
	}
    else if (turn_dir > 0)
	{
	if (avoidance_heading_direction < 0)
	    {
	    avoidance_heading_direction = avoidance_heading_direction * -1.f; // flip direction
	    }
	}
    // if turn_dir == 0, the heading does not need to change

    return;
    }
