/*
  * Copyright (C) Kevin van Hecke
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
 * @file "modules/vision_outback/vision_outback.h"
 * @author Kevin van Hecke
 * Vision uart (RS232) communication
 */

#ifndef VISION_OUTBACK_H
#define VISION_OUTBACK_H

#include "std.h"
#include "generated/airframe.h"
#include "pprzlink/pprz_transport.h"
#include "math/pprz_orientation_conversion.h"
#include "math/pprz_algebra_float.h"


/* Main vision_outback structure */
struct vision_outback_t {
  struct link_device *device;           ///< The device which is uses for communication
  struct pprz_transport transport;      ///< The transport layer (PPRZ)
  struct OrientationReps imu_to_mag;    ///< IMU to magneto translation
  bool msg_available;                 ///< If we received a message
};


//should be exactly the same as pprz.h
struct Vision2PPRZPackage {
    int32_t frame_id;
    float height;
    float out_of_range_since;
    float marker_enu_x;
    float marker_enu_y;
    float land_enu_x;
    float land_enu_y;
    float flow_x;
    float flow_y;
    uint8_t status;
} __attribute__((__packed__));
extern struct Vision2PPRZPackage v2p_package;

//should be exactly the same as pprz.h
struct PPRZ2VisionPackage {
    float qi;
    float qx;
    float qy;
    float qz;
    float gpsx;
    float gpsy;
    float gpsz;
    float geo_init_gpsx;
    float geo_init_gpsy;
    float geo_init_gpsz;
    unsigned char enables;
}__attribute__((__packed__));

extern float vision_outback_search_height;
extern float vision_outback_moment_height;
extern bool vision_outback_enable_landing ;
extern bool vision_outback_enable_take_foto;
extern bool vision_outback_enable_findjoe;
extern bool vision_outback_enable_opticflow;
extern bool vision_outback_enable_attcalib;
extern bool vision_outback_enable_videorecord;
extern bool vision_outback_shutdown;
extern struct FloatVect3 land_cmd;
extern bool het_moment;
extern bool vision_timeout;

extern void vision_outback_init(void);
extern void vision_outback_event(void);
extern void vision_outback_periodic(void);

extern void enableVisionLandingspotSearch(bool b);
extern void enableVisionDescent(bool b);
extern void enableVisionOpticFlow(bool b);
extern void enableVisionFindJoe(bool b);
extern bool enableVisionAttCalib(bool b);
extern bool enableVisionVideoRecord(bool b);
extern bool enableVisionShutdown(bool b);

extern bool getVisionReady(void);

#endif
