//
// Created by michiel on 20-03-21.
//

#include "cv_optical_flow.h"
#include "modules/computer_vision/cv_detect_color_object.h"
#include "modules/computer_vision/cv.h"
#include "subsystems/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

#define PRINT(string, ...) fprintf(stderr, "[contour_detector->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)

static pthread_mutex_t mutex;

/*
 * object_detector
 * @param img - input image to process
 * @param filter - which detection filter to process
 * @return img
 */
static struct image_t *object_detector(struct image_t *img, uint8_t filter)
{
    // Optical flow processing code goes here //

    return img;
}

struct image_t *optical_flow(struct image_t *img);
struct image_t *optical_flow(struct image_t *img)
{
    return object_detector(img, 1);
}

void optical_flow_detector_init(void) {
    memset(global_filters, 0, 2 * sizeof(struct color_object_t));
    pthread_mutex_init(&mutex, NULL);
#ifdef COLOR_OBJECT_DETECTOR_CAMERA1

    // Register optical flow processing to main camera
    cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, optical_flow, 0);
#endif
}

void optical_flow_periodic(void) {
    static struct color_object_t local_filters[2];
    pthread_mutex_lock(&mutex);
    memcpy(local_filters, global_filters, 2 * sizeof(struct color_object_t));
    pthread_mutex_unlock(&mutex);

    if (local_filters[0].updated) {
        AbiSendMsgOPTICAL_FLOW(COLOR_OBJECT_DETECTION1_ID, local_filters[0].x_c, local_filters[0].y_c,
                                   0, 0, local_filters[0].color_count, turn);
        local_filters[0].updated = false;
    }
    if (local_filters[1].updated) {
        AbiSendMsgOPTICAL_FLOW(COLOR_OBJECT_DETECTION2_ID, local_filters[1].x_c, local_filters[1].y_c,
                                   0, 0, local_filters[1].color_count, 1);
        local_filters[1].updated = false;
    }
}