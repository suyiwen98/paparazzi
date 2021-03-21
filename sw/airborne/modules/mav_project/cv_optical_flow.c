//
// Created by michiel on 20-03-21.
// Changed by Marko on 20-03-21
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
    int32_t foe_x; // Focus of expansion
    int32_t foe_y;
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
static struct image_t *optical_flow_detector(struct image_t *img) {
    // Optical flow processing code goes here //

    // convert the images to grayscale and blur it to remove noise
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.blur(prev_gray, (3, 3))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))

    // initial threshold
    thresh = 20

    // Pole Detector
    Filtered = filter_color(prev_bgr, y_low=50, y_high=250, u_low=0, u_high=150, v_low=140, v_high=220)

    // Reduce noise on detected pole (only pole remains, smaller surface areas with orange pixels are not considered)
	
    Filtered = np.uint8(Filtered)
    reduced_noise = cv2.fastNlMeansDenoisingColored(Filtered, None, 90, 10, 7, 21)
    ret, reduced_noise = cv2.threshold(reduced_noise, 250, 255, cv2.THRESH_BINARY)

    // Find the contours and rectangular areas of interest
    img_contoured, boundRect, front_contours = get_front_contour(reduced_noise, thresh)

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

// Added by Marko on 20-03-21
// Based on extract_information_flow_field.py, lines 150-174
// NOTE: PLEASE HAVE A LOOK into the Python code if the unfinished C/Python code confuses you

/* Functions filters the image based on YUV thresholds and returns binary filter inputs: 
im: bgr colored image 
y_low,y_high, u_low, u_high, v_low, v_high: YUV thresholds
*/

struct image_t * filter_color(struct image_t * img_in, struct image_t * img_out, uint8_t y_low, uint8_t y_high, uint8_t u_low, uint8_t u_high, uint8_t v_low, uint8_t v_high)
	{
	// Convert an image from RBG space to YUV
	cvtColor(img_in, img_out, CV_BGR2YUV);
	
    YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    Filtered = np.zeros([YUV.shape[0], YUV.shape[1], 3])
	
    for y in range(YUV.shape[0]):
        for x in range(YUV.shape[1]):
            if (y_low <= YUV[y, x, 0] <= y_high && u_low <= YUV[y, x, 1] <= u_high && v_low <= YUV[y, x, 2] <= v_high):
                Filtered[y, x, 0] = 1
                Filtered[y, x, 1] = 1
                Filtered[y, x, 2] = 1

    // Make the Image Binary
	struct image_t * img_intermediate = img_out
	
    ret, Filtered = cv2.threshold(Filtered, 0, 255, cv2.THRESH_BINARY)
	
	// img_intermediate and img_out point at the same memory, just introduced the intermediate variable for readibility, is that possible?
	threshold(img_intermediate, img_out, 0, 255, TRESH_BINARY);

    return img_out
	}
	
// Based on extract_information_flow_field.py, lines 76-126
// NOTE: PLEASE HAVE A LOOK into the Python code if the unfinished C/Python code confuses you
// Probably needs to return a pointer to an integer array later
int * get_front_contour(struct image_t * gray, uint8_t val)
	{	
    /*Inputs:
        gray: grayscale image
        val: threshold value
		
    Outputs:
        Contoured: image with all contours
        boundRect: bounding rectangular parameters (x of top left corner, y, width, height)
        c: coordinates of contour with maximum height
		*/

    threshold = val
    //Detect edges using Canny
    canny_output = cv2.Canny(gray, threshold, threshold * 2)

    //Find contours
    _, contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    // Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    boundRect = [None] * len(contours)
    h = [None] * len(contours)
    for i in range(len(contours)):
        // approximate the contour
        peri = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.01 * peri, True)
        boundRect[i] = cv2.boundingRect(approx)  # (x,y,w,h)
        h[i] = boundRect[i][3]
        // random color for each contour
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
                      (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

    // get contours of maximum height
    front_contours = contours[np.argmax(h)]
	// bounded rectable of the contour with max heigt
    boundRect = cv2.boundingRect(front_contours)
    // Change contour colors to Blue
    drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)

    // Make the Image Binary
    ret, drawing = cv2.threshold(drawing, 1, 255, cv2.THRESH_BINARY)
    // Show in a window
    kernel = np.ones((5, 5), np.uint8)
	
	// erosion = cv2.erode(drawing, kernel, iterations=1)
    closing = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, kernel)
    Contoured = closing
    return Contoured, boundRect, front_contours	
	}


struct image_t *optical_flow(struct image_t *img);

struct image_t *optical_flow(struct image_t *img) {
    return optical_flow_detector(img);
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