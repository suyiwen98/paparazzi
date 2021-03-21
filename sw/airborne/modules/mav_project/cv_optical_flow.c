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

#define PRINT(string, ...) fprintf(stderr, "[optical_flow->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)

static pthread_mutex_t mutex;

// define global variables
struct optical_flow_object_t {
    int32_t error;
    int32_t foe; // Focus of expansion
    int32_t ttc; // Time to contact
    int32_t divergence;
    bool updated;
};
struct optical_flow_object_t global_filters[1];

/*
 * object_detector
 * @param img - input image to process
 * @param filter - which detection filter to process
 * @return img
 */
static struct image_t *optical_flow_detector(struct image_t *img)
{
    // Optical flow processing code goes here //

    // Set variables to global filters to be sent in a message
    // Apparently we only need ttc, so the rest could be removed if they are a hassle to add
    pthread_mutex_lock(&mutex);
    global_filters[0].error = error;
    global_filters[0].foe = foe;
    global_filters[0].ttc = ttc;
    global_filters[0].divergence = divergence;
    global_filters[0].updated = true;
    pthread_mutex_unlock(&mutex);
    return img;
}

struct image_t *optical_flow(struct image_t *img);
struct image_t *optical_flow(struct image_t *img)
{
    return optical_flow_detector(img, 1);
}

void optical_flow_init(void) {
    memset(global_filters, 0, 2 * sizeof(struct optical_flow_object_t));
    pthread_mutex_init(&mutex, NULL);
#ifdef OPTICAL_FLOW_DETECTOR_CAMERA1

    // Register optical flow processing to main camera
    cv_add_to_device(&OPTICAL_FLOW_DETECTOR_CAMERA1, optical_flow, 0);
#endif
}

void optical_flow_periodic(void) {
    static struct optical_flow_object_t local_filters[2];
    pthread_mutex_lock(&mutex);
    memcpy(local_filters, global_filters, 2 * sizeof(struct optical_flow_object_t));
    pthread_mutex_unlock(&mutex);

    if (local_filters[0].updated) {
        AbiSendMsgOPTICAL_FLOW(OPTICAL_FLOW_DETECTION1_ID, local_filters[0].error, local_filters[0].foe,
                                   local_filters[0].ttc, local_filters[0].divergence);
        local_filters[0].updated = false;
    }
}