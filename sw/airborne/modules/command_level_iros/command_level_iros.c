/*
 * Copyright (C) Shuo Li
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
 * @file "modules/command_level_iros/command_level_iros.c"
 * @author Shuo Li
 * This module is the highest level
 */

#include <stdio.h>
#include "modules/command_level_iros/command_level_iros.h"
#include "firmwares/rotorcraft/autopilot.h"
#include "modules/flight_plan_in_guided_mode/flight_plan_in_guided_mode.h"
#include "modules/flight_plan_in_guided_mode/flight_plan_clock.h"
#include "modules/state_autonomous_race/state_autonomous_race.h"
#include "modules/computer_vision/snake_gate_detection.h"
#include "modules/kalman_filter/kalman_filter.h"

#define PI 3.1415926

uint8_t previous_mode;
uint8_t current_mode;
uint8_t previous_lower_level;

double theta_bias;
double phi_bias;
double theta_hover;
double phi_hover;
struct acceleration accel_bias;
struct acceleration accel_hover;
double desired_theta;
double desired_phi;

void first_part_logic(void);
void second_part_logic(void);
void third_part_logic(void);
void fourth_part_logic(void);
void fifth_part_logic(void);

//bool replay_flag;
bool flag_init_geo;
enum states_lower_level state_lower_level = WAIT_FOR_DETECTION_CM;
enum states_upper_level state_upper_level = SECOND_PART;



//temp fix for data set recording
int init_pos_filter = 0;


void command_init(){
    previous_mode = autopilot_mode;
    current_mode = autopilot_mode;
    init_heading = stateGetNedToBodyEulers_f()->psi;
}


void command_run() {
  
    current_mode = autopilot_mode;
    if (previous_mode != current_mode)
    {
        counter_autopilot_mode = 0;
        time_autopilot_mode = 0;
        primitive_in_use = NO_PRIMITIVE;
        state_lower_level = PREPARE_CM; //PREPARE_CM;
        state_upper_level = FIRST_PART;
        init_heading = stateGetNedToBodyEulers_f()->psi;
		flag_init_geo = FALSE;
    }
    if (autopilot_mode != AP_MODE_MODULE) {
        return;
    }

    if(state_upper_level  == FIRST_PART)
    {

        first_part_logic();
    }

    if(state_upper_level  == SECOND_PART)
    {
        second_part_logic();
    }

    if(state_upper_level  == THIRD_PART)
    {
        third_part_logic();
    }

    if(state_upper_level  == FOURTH_PART)
    {
        fourth_part_logic();
    }

    if(state_upper_level  == FIFTH_PART)
    {
			fifth_part_logic();
    }
    previous_mode = current_mode;
}








void first_part_logic()
{
    switch(state_lower_level)
    {
        case PREPARE_CM:
			calculate_attitude_average(&theta_bias,&phi_bias,&accel_bias);
			if (sample_pointer == SAMPLE_NUM)
            {
                previous_lower_level = PREPARE_CM;
                state_lower_level = TAKE_OFF_OPEN_LOOP_CM;
            }
            break;
			
        case TAKE_OFF_OPEN_LOOP_CM:
            take_off(TAKE_OFF_ALTITUDE);
            if(states_race.altitude_is_achieved == TRUE)
            {
                previous_lower_level = TAKE_OFF_OPEN_LOOP_CM;
                state_lower_level =  TAKE_OFF_CLOSE_LOOP_CM;
            }
            break;

        case TAKE_OFF_CLOSE_LOOP_CM:
            hold_altitude(TAKE_OFF_ALTITUDE);
            if(states_race.altitude_is_achieved == TRUE)
            {
                previous_lower_level = TAKE_OFF_CLOSE_LOOP_CM;
                state_lower_level = HOVER_CM;
            }
            break;

        case HOVER_CM:
            hover();
			/*calculate_attitude_average(&theta_hover,&phi_hover,&accel_hover);*/
					/*if (sample_pointer == SAMPLE_NUM)*/
					if (1)
					{
							previous_mode = HOVER_CM;
							state_lower_level = APPROACH_GATE_CM;
							state_upper_level = SECOND_PART;
							counter_temp3 = 0;
							time_temp3 = 0;
							/*file_logger_start();*/
					}
            break;

    }
}



void second_part_logic()
{
	switch(state_lower_level)
	{
			case APPROACH_GATE_CM:
					arc_open_loop(1.0,-10.0/180*3.14);
	}
}

void third_part_logic()
{

}


void fourth_part_logic() {
}


void fifth_part_logic()
{
}