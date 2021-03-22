//
// Created by michiel on 20-03-21.
// Changed by Marko on 20-03-21
//

#include "modules/computer_vision/cv.h"
#include "cv_optical_flow.h"
#include "opencv_optical_flow.h"

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
    int32_t foe_x; // Focus of expansion
    int32_t foe_y;
    int32_t ttc; // Time to contact
    int32_t divergence;
    bool updated;
};
struct optical_flow_object_t global_filters[1];


struct image_t *optical_flow(struct image_t *img);
struct image_t *optical_flow(struct image_t *img) {
    // Optical flow processing code goes here //
    //TODO: Actually send img_prev and not just the same image twice
    char* result = opencv_optical_flow((char *) img->buf, img->w, img->h, (char *) img->buf, img->w, img->h);
    PRINT("Got result %d\n", result);
//    img->buf = result; // This line will crash the program

    // Set some dummy values for testing
    int32_t error = 1;
    int32_t foe_x = 2;
    int32_t foe_y = 3;
    int32_t ttc = 12;
    int32_t divergence = 9;
    // Set variables to global filters to be sent in a message
    // Apparently we only need ttc, so the rest could be removed if they are a hassle to add
    pthread_mutex_lock(&mutex);
    global_filters[0].error = error;
    global_filters[0].foe_x = foe_x;
    global_filters[0].foe_y = foe_y;
    global_filters[0].ttc = ttc;
    global_filters[0].divergence = divergence;
    global_filters[0].updated = true;
    pthread_mutex_unlock(&mutex);
    return img;
}

void optical_flow_init(void) {
    memset(global_filters, 0, sizeof(struct optical_flow_object_t));
    pthread_mutex_init(&mutex, NULL);
#ifdef OPTICAL_FLOW_DETECTOR_CAMERA1

    // Register optical flow processing to main camera
    cv_add_to_device(&OPTICAL_FLOW_DETECTOR_CAMERA1, optical_flow, 0);
#endif
}

void optical_flow_periodic(void) {
    static struct optical_flow_object_t local_filters[2];
    pthread_mutex_lock(&mutex);
    memcpy(local_filters, global_filters, sizeof(struct optical_flow_object_t));
    pthread_mutex_unlock(&mutex);

    if (local_filters[0].updated) {
        AbiSendMsgOPTICAL_FLOW2(OPTICAL_FLOW_ID,
                                local_filters[0].error,
                                local_filters[0].foe_x,
                                local_filters[0].foe_y,
                                local_filters[0].ttc,
                                local_filters[0].divergence);
        local_filters[0].updated = false;
    }
}