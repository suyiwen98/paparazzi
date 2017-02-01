/*
 * Copyright (C) Kirk Scheper
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
 * @file "modules/droplet/droplet.c"
 * @author Kirk Scheper
 * Droplet collision droplet avoidance system
 */

#include "modules/droplet/droplet.h"
#include "mcu_periph/sys_time.h"
#include "modules/stereocam/stereocam2state/stereocam2state.h"
#include "subsystems/radio_control.h"
#include "firmwares/rotorcraft/stabilization/stabilization_attitude_rc_setpoint.h"

#ifndef DELFLY_SENDER_ID
#define DELFLY_SENDER_ID ABI_BROADCAST
#endif

enum {
 DROPLET_UNOBSTRUCTED,
 DROPLET_WAIT_AFTER_DETECTION,
 DROPLET_AVOID,
 DROPLET_UNDEFINED
};

// parameter setting
uint16_t obst_thr_1 = 150;//7;      // obstacle threshold for phase 1
uint16_t disp_thr_1 = 5;//10;     // obstacle count minimum threshold for phase 1
uint32_t obst_wait_2 = 3000;  // -->1800<-- wait time for phase 2
uint16_t obst_thr_3 = 40;     // obstacle threshold for phase 3
uint16_t obst_thr_4 = 40;     // obstacle threshold for phase 4
uint16_t obst_wait_4 = 500;   // wait time for phase 4

uint16_t obst_cons_1 = 3;     // obstacle consistency threshold for phase 1
uint16_t obst_cons_3 = 1;     // no-obstacle consistency threshold for phase 3
uint16_t obst_cons_5 = 1;     // obstacle consistency threshold for phase 4

// init
uint16_t obst_count_1 = 0;    // counter for sequential obstacle detections in phase 1
uint16_t obst_free_3 = 0;     // counter for sequential no-obstacle detections in phase 3
uint16_t obst_dect_4 = 0;     // counter for sequential obstacle detections in phase 4
float obst_time = 0;       // timer for phase 2 and 4

uint8_t droplet_state = DROPLET_UNDEFINED;
uint32_t prev_time = 0;

float wall_following_trim = 0;   // yaw rate trim to force vehicle to follow wall
int16_t turn_direction = 1;

uint8_t droplet_active = 1;

void droplet_init(void)
{
}

/* Set turn command based on current droplet state
 *
 */
void droplet_periodic(void){
  if (droplet_active){return;}
  if (radio_control.values[5] < 0){nus_turn_cmd = 0; return;} // this should be ELEV D/R

  switch(droplet_state){
    case DROPLET_UNOBSTRUCTED:
      nus_turn_cmd = (int16_t)(wall_following_trim * turn_direction * MAX_PPRZ / STABILIZATION_ATTITUDE_SP_MAX_R); // go slight left to follow wall
      break;
    case DROPLET_WAIT_AFTER_DETECTION:
      nus_turn_cmd = 0; // go straight
      break;
    case DROPLET_AVOID:
      nus_turn_cmd =  1 * turn_direction * MAX_PPRZ / STABILIZATION_ATTITUDE_SP_MAX_R; // avoid right with 1. m/s
      break;
    case DROPLET_UNDEFINED:
      nus_turn_cmd = 0; // go straight until we are sure there are no obstacles
      break;
    default:
      break;
  }
}

#include "subsystems/abi.h"
#include "autopilot.h"
void delfly_vel_estimator(void)
{
  if(!autopilot_in_flight)
    return;

  // body to speed 0.019272*deg - 0.039499
  float speed = 1.10376*stateGetNedToBodyEulers_f()->theta - 0.039499;

  AbiSendMsgVELOCITY_ESTIMATE(DELFLY_SENDER_ID, get_sys_time_usec(),
      speed * sin(stateGetNedToBodyEulers_f()->theta), 0,
      speed * cos(stateGetNedToBodyEulers_f()->theta), 0.3);
}

/* runs a droplet_state machine to implement the droplet
 * droplet_state 1: DROPLET_UNOBSTRUCTED flight
 * droplet_state 2: obstacle detected, wait for action
 * droplet_state 3: avoid
 * droplet_state 4: fly straight, but be aware of undetected obstacles
 */
void run_droplet(uint32_t disparities_total, uint32_t disparities_high)
{
  // Control logic
  switch(droplet_state){
    case DROPLET_UNOBSTRUCTED: // DROPLET_UNOBSTRUCTED flight
      if (disparities_high > obst_thr_1 || disparities_total < disp_thr_1) { // if true, obstacle in sight
        obst_count_1++;
      } else if (obst_count_1 > 0) {
        obst_count_1--;
      }

      if (obst_count_1 > obst_cons_1) { // if true, obstacle is consistent
        droplet_state = DROPLET_WAIT_AFTER_DETECTION;
        obst_count_1 = 0; // set zero for later
        obst_time = get_sys_time_float();
      }
      break;
    case DROPLET_WAIT_AFTER_DETECTION: // obstacle detected, wait for action
      if (1000*(get_sys_time_float() - obst_time) > obst_wait_2) {
        droplet_state = DROPLET_AVOID;
      }
      break;
    case DROPLET_AVOID: // avoid
      if (disparities_high < obst_thr_3 && disparities_total > disp_thr_1) { // if true, flight direction is safe
        obst_free_3++;
      } else {
        obst_free_3 = 0;
      }

      if (obst_free_3 > obst_cons_3) { // if true, consistently no obstacles
        droplet_state = DROPLET_UNDEFINED;
        obst_free_3 = 0; // set zero for later
        obst_time = get_sys_time_float();
      }
      break;
    case DROPLET_UNDEFINED: // fly straight, but be aware of undetected obstacles
      if (disparities_high > obst_thr_4 || disparities_total < disp_thr_1) { // if true, obstacle in sight
        obst_dect_4++;
      } else {
        obst_dect_4 = 0;
      }

      if (obst_dect_4 > obst_cons_5) { // if true, obstacle is consistent
        droplet_state = DROPLET_AVOID; // go back to droplet_state 3
        obst_dect_4 = 0; // set zero for later
      } else if (1000*(get_sys_time_float() - obst_time) > obst_wait_4) {
        droplet_state = DROPLET_UNOBSTRUCTED;
        obst_dect_4 = 0;
      }
      break;
    default:
      break;
  }
}

/* runs a droplet_state machine to implement the droplet
 * droplet_state 1: DROPLET_UNOBSTRUCTED flight
 * droplet_state 3: avoid
 * droplet_state 4: fly straight, but be aware of undetected obstacles
 */
void run_droplet_low_texture(uint32_t disparities_high, uint32_t disparities_total, uint32_t histogram_obs,
    uint32_t count_disps_left, uint32_t count_disps_right)
{
  nus_climb_cmd = disparities_high;
  if (histogram_obs/10 > 127){
    nus_gate_heading = 127;
  } else {
    nus_gate_heading = histogram_obs/10;
  }

  // => max_Y = 500 mm
  // 35 too sensitive, many turns
  // 50 little bit less sensitive, but when approaching rich textured wall, still turns too early (cyberzoo sides also detected!)
  // 60 seems to work better, richly textured stuff is approached more closely. Orange poles are also detected. crashes into the net when background is blue doors

  // => max_Y = 400 mm
  // 60 works perfectly if sufficient space is available
  // 40 gives better performance with orange pole in middle of the space
  // 5 works for pole in middle

  // Control logic
  switch(droplet_state){
    case DROPLET_UNOBSTRUCTED: // unobstructed flight
      if (histogram_obs > obst_thr_1 || count_disps_left < disp_thr_1 || count_disps_right < disp_thr_1 ) { // if true, obstacle in sight
        droplet_state = DROPLET_AVOID;
      }
      break;
    case DROPLET_AVOID: // avoid
      if (disparities_high < obst_thr_3  && count_disps_left > disp_thr_1 && count_disps_right > disp_thr_1) { // if true, flight direction is safe
        obst_free_3++;
      } else {
        obst_free_3 = 0;
      }

      if (obst_free_3 > obst_cons_3) { // if true, consistently no obstacles
        droplet_state = DROPLET_UNDEFINED;
        obst_free_3 = 0; // set zero for later
        obst_time = get_sys_time_float();
      }
      break;
    case DROPLET_UNDEFINED: // fly straight, but be aware of undetected obstacles
      if (disparities_high > obst_thr_4 || disparities_total < disp_thr_1) { // if true, obstacle in sight
        obst_dect_4++;
      } else {
        obst_dect_4 = 0;
      }

      if (obst_dect_4 > obst_cons_5) { // if true, obstacle is consistent
        droplet_state = DROPLET_AVOID; // go back to phase 3
        obst_dect_4 = 0; // set zero for later
      } else if (1000*(get_sys_time_float() - obst_time) > obst_wait_4) {
        droplet_state = DROPLET_UNOBSTRUCTED;
        obst_dect_4 = 0;
      }
      break;
    default:
      break;
  }
}

#include "subsystems/datalink/telemetry.h"
static float dist2left = 0, dist2right = 0, ang2left = 0, ang2right = 0;
void wall_estimate(float slope_l, float intercept_l, float fit_l, float slope_r, float intercept_r, float fit_r)
{
  float alpha = 0.3;
  if (fit_l < INT32_MAX && abs(slope_l) > 0.001) {
    dist2left += alpha*(-(float)intercept_l/slope_l - dist2left);
    ang2left += alpha*(1.57 - atan(100/slope_l) - ang2left);
  }
  if (fit_r < INT32_MAX && abs(slope_l) > 0.001) {
    dist2right += alpha*(-(float)intercept_r/slope_r - dist2right);
    ang2right += alpha*(1.57 - atan(100/slope_r) - ang2right);
  }
  float dummyf;
  uint8_t dummy8;
  uint16_t dummy16;
  DOWNLINK_SEND_RANGEFINDER(DefaultChannel, DefaultDevice, &dummy16, &dist2left, &ang2left, &dist2right, &ang2right, &dummyf, &dummy8);
}