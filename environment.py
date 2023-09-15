# Date: 8/3/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...
import random
import sys

from nearest_location import nearest_location
from consumption_duration import consumption_duration
from consumption_duration import haversine

import math
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.envs.classic_control import utils
from gymnasium import logger, spaces


# The environment of route planning,
#  An electric vehicles need to choose the next action according to the current state
#  An agent needs to choose the next action according to the current state, and observes the next state and reward


## Action Space is a tuple, with 3 arrays, Action a:=(next_node, charge, rest )
# next_node:= {1,2,3,4,5}
# charge:= {0, 0.3, 0.5, 0.8}
# rest:= {0, 0.3, 0.6, 0.9, 1}


## State/Observation Space is a np.ndarray, with shape `(8,)`,
# State s := (current_node, x1, y1, soc,t_stay, t_secd, t_secr, t_secch)



# Rewards
#r1: Reward for the distance to the target
#r2: Reward based on Battery’s operation limits
#r3: Reward for the suitable charging time
#r4: Reward for the suitable driving time
#r5: Reward for the suitable rest time at parking lots


# Starting State
# random state


# Episode End
# The episode ends if any one of the following occurs:
# 1. Trapped on the road
# 2. Many times SoC is less than 0.1 or greater than 0.8, which violates the energy constraint
# 3. t_secd is greater than 4.5, which violates the time constraint
# 4. Episode length is greater than 500
# 5. Reach the target

class rp_env(gym.Env[np.ndarray, np.ndarray]):

    def __init__(self, render_mode: Optional[str] = None):
        #initialization
        self.x_source = 52.66181 #source
        self.y_source = 13.38251
        self.c_source = 47
        self.x_target = 51.772324 #target
        self.y_target = 12.402652
        self.c_target = 88
        self.m = 13500 #(Leergewicht) in kg
        self.g = 9.81
        self.rho = 1.225
        self.A_front = 10.03
        self.c_r = 0.01
        self.c_d = 0.7
        self.a = 0
        self.eta_m = 0.82
        self.eta_battery = 0.82

        self.file_path_ch = 'cs_combo_bbox.csv'
        self.file_path_p = 'parking_bbox.csv'
        self.data_ch = pd.read_csv("cs_combo_bbox.csv")
        self.data_p = pd.read_csv("parking_bbox.csv")
        self.n_ch = 3 # Number of nearest charging station
        self.n_p =2 # Number of nearest parking lots


        # Limitation of battery
        self.battery_capacity = 588 #(in kWh)
        self.soc_min = 0.1
        self.soc_max = 0.8
        # Each section has the same fixed travel time
        self.min_rest = 2700 # in s
        self.max_driving = 16200 # in s
        self.section = self.min_rest + self.max_driving

        self.w_distance = 1
        self.w_trapped = 1
        self.w_charge = 1
        self.w_rest = 1
        self.w_driving = 1

        self.num_trapped = 0 # The number that trapped on the road
        self.max_trapped = 10

        # # Initialize the actoin space, state space
        self.action_space = spaces.Discrete(22)
        self.df_actions = pd.read_csv("actions.csv")
        
        self.state = None

    def cs_elevation_power(self,x1, y1):
        matching_row = self.data_ch[(self.data_ch["Latitude"] == x1) & (self.data_ch["Longitude"] == y1)]
        if not matching_row.empty:
            cs_elevation = matching_row["Elevation"].values[0]
            cs_power = matching_row["Power"].values[0]
        else:
            print("Current location not found in the dataset of cd")

        return(cs_elevation, cs_power)

    def p_elevation(self, x1, y1):
        matching_row = self.data_p[(self.data_p["Latitude"] == x1) & (self.data_p["Longitude"] == y1)]
        if not matching_row.empty:
            p_elevation = matching_row["Altitude"].values[0]
        else:
            print("Current location not found in the dataset of p")

        return(p_elevation)


    def step(self, action):
        #Run one timestep of the environment’s dynamics using the agent actions.
        #Calculate reward, update state
        #At the end of an episode, call reset() to reset this environment’s state for the next episode.


        terminated = False
        #Check if the action is valid
        # assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        #Obtain current state
        # If current POI is a charging station, soc is battery capacity that after charging, t_secch_current includes charging time at the current location
        # If current POI is a parking lot, t_secp_current includes rest  time at the current location
        node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = self.state
        # print(node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current) #test

        #Obtain selected action
        index_cpu = action.cpu()
        next_node, charge, rest = self.df_actions.iloc[index_cpu.item()]
        print('next_node, charge, rest = ', next_node, charge, rest)

        # Obtain the altitude and/or power of current location
        if node_current in [1, 2, 3]:  # charging station
            c_current, power_current = self.cs_elevation_power(x_current, y_current)
        else:  # parking lots
            c_current = self.p_elevation(x_current, y_current)

        # Obtain n nearest POIs
        nearest_ch = nearest_location(self.file_path_ch, x_current, y_current, self.n_ch)
        # print(nearest_ch)
        nearest_x1 = nearest_ch.loc[0, 'Latitude']
        nearest_y1 = nearest_ch.loc[0, 'Longitude']
        nearest_x2 = nearest_ch.loc[1, 'Latitude']
        nearest_y2 = nearest_ch.loc[1, 'Longitude']
        nearest_x3 = nearest_ch.loc[2, 'Latitude']
        nearest_y3 = nearest_ch.loc[2, 'Longitude']
        # print('nearest_1-3:', nearest_x1, nearest_y1, nearest_x2, nearest_y2, nearest_x3, nearest_y3)

        nearest_p = nearest_location(self.file_path_p, x_current, y_current, self.n_p)
        # print(nearest_p)
        nearest_x4 = nearest_p.loc[0, 'Latitude']
        nearest_y4 = nearest_p.loc[0, 'Longitude']
        nearest_x5 = nearest_p.loc[1, 'Latitude']
        nearest_y5 = nearest_p.loc[1, 'Longitude']
        # print('nearest_4-5:', nearest_x4, nearest_y4, nearest_x5, nearest_y5)

        # Obtain the energy and time consumption from current node to next node
        if next_node == 1:
            d_next = haversine(nearest_x1, nearest_y1, self.x_target, self.y_target)
            #consumption and typical_duration from current location to next node
            nearest_c1, next_power = self.cs_elevation_power(nearest_x1, nearest_y1)
            consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, c_current, nearest_x1, nearest_y1,
                                                                                nearest_c1, self.m, self.g,
                                                                                self.c_r, self.rho, self.A_front,
                                                                                self.c_d, self.a, self.eta_m,
                                                                                self.eta_battery)
        if next_node == 2:
            d_next = haversine(nearest_x2, nearest_y2, self.x_target, self.y_target)
            nearest_c2, next_power = self.cs_elevation_power(nearest_x2, nearest_y2)
            consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, c_current,
                                                                                nearest_x2, nearest_y2,
                                                                                nearest_c2, self.m, self.g,
                                                                                self.c_r, self.rho, self.A_front,
                                                                                self.c_d, self.a, self.eta_m,
                                                                                self.eta_battery)
        if next_node == 3:
            d_next = haversine(nearest_x3, nearest_y3, self.x_target, self.y_target)
            nearest_c3, next_power = self.cs_elevation_power(nearest_x3, nearest_y3)
            consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, c_current,
                                                                                nearest_x3, nearest_y3,
                                                                                nearest_c3, self.m, self.g,
                                                                                self.c_r, self.rho, self.A_front,
                                                                                self.c_d, self.a, self.eta_m,
                                                                                self.eta_battery)
        if next_node == 4:
            d_next = haversine(nearest_x4, nearest_y4, self.x_target, self.y_target)
            nearest_c4 = self.p_elevation(nearest_x4, nearest_y4)
            consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, c_current,
                                                                                nearest_x4, nearest_y4,
                                                                                nearest_c4, self.m, self.g,
                                                                                self.c_r, self.rho, self.A_front,
                                                                                self.c_d, self.a, self.eta_m,
                                                                                self.eta_battery)
        if next_node == 5:
            d_next = haversine(nearest_x5, nearest_y5, self.x_target, self.y_target)
            nearest_c5 = self.p_elevation(nearest_x5, nearest_y5)
            consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, c_current,
                                                                                nearest_x5, nearest_y5,
                                                                                nearest_c5, self.m, self.g,
                                                                                self.c_r, self.rho, self.A_front,
                                                                                self.c_d, self.a, self.eta_m,
                                                                                self.eta_battery)
        print("Length, speed, consumption", length_meters/1000, "km", length_meters/typical_duration * 3.6, "km/h", consumption/length_meters*100000, "kWh/100km\n")
        
        # the distance from current location to target
        d_current = haversine(x_current, y_current, self.x_target, self.y_target)
        # soc after driving
        soc_after_driving = soc - consumption / self.battery_capacity
        # the time that arriving next location
        t_arrival = t_secd_current + t_secch_current + t_secp_current + typical_duration
        # the driving time when arrive next location
        t_secd_current = t_secd_current + typical_duration
             
        # Calculate reward for distance
        if d_next == 0:
            r_distance = 10
            terminated = True
            print("Terminated: Arrival target")
        else:
            # r_distance = np.exp * ((d_current - d_next) / 25000) - 1
            r_distance = (d_current - d_next) / 25000
            
        # Reward for battery      
        # If there is recuperated energy, the soc can be charged up to 0.8
        if consumption < 0:
            if soc_after_driving > 0.8:
                soc_after_driving = 0.8

        # Punishment for the trapped on the road
        if soc_after_driving < 0: # Trapped
            terminated = True
            r_energy = - 6
            print("Terminated: Trapped on the road, should be reseted")
        else: # No trapped
            if soc_after_driving < 0.1: # Still can run, but violated constraint
                #r_energy =  np.log(self.w3 * abs(soc_after_driving - 0.1))
                r_energy = np.log(0.1 * soc_after_driving) + 5
                self.num_trapped += 1
                if self.num_trapped == 10:
                    terminated = True # Violate the self.max_trapped times, stop current episode
                    print("Terminated: Violated soc 10 times,should be reseted")
            else:
                r_energy = 0.4 # No trapped

        # Calculate reward for suitable driving time when arriving next node  
        # update t_secd_current
        if t_arrival >= self.section: # A new section begin before arrival next state
            t_secd_current = t_arrival % self.section
            rest_time = t_secp_current + t_secch_current        
            if rest_time < self.min_rest:
                terminated = True
                print("Terminated: Violated self.max_driving times,should be reseted")
                if (self.section - rest_time - self.max_driving) < 0:
                    print("Warning! wrong Value of rest time")
                    sys.exit(1)
                else:
                    # r_driving = -10 * (self.section - rest_time - self.max_driving)
                    r_driving = -100
            else:
                r_driving = np.exp((self.section - rest_time)/3600) - np.exp(4.5)
        else: # still in current section when arriving next poi
            if t_secd_current <= self.max_driving:
                r_driving = np.exp(t_secd_current / 3600) - np.exp(4.5)
            else:
                print("Terminated: Violated self.max_driving times,should be reseted")
                terminated = True
                # r_driving = -10 * (t_secd_current- self.max_driving)
                r_driving = -100

        #next node is an charging station
        # update t_stay, t_secch_current,t_secp_current
        if next_node in [1, 2, 3]:
            if charge == 0:
                r_charge = 0
                t_stay = 0
                if t_arrival >= self.section:  # A new section begin before arrival next state
                    t_secp_current = 0
                    t_secch_current = 0
            else:
                if charge >= soc_after_driving:# Calculate reward for suitable charging time in next node
                    t_stay = (charge - soc_after_driving) * self.battery_capacity / next_power * 3600 #in s
                    t_departure = t_arrival + t_stay
                    if t_arrival >= self.section:  # A new section begin before arrival next state
                        t_secp_current = 0
                        t_secch_current = t_stay
                        if t_secch_current < self.min_rest:
                            r_charge = np.exp(5 * t_secch_current / 3600) - np.exp(3.75)
                        else:
                            # r_charge = -10 * (np.exp(1.5 * t_secch_current / 3600) - np.exp(1.125))
                            r_charge = -32 * t_secch_current / 3600 + 24
                    else:
                        if t_departure >= self.section:  # A new section begin before leaving next state
                            t_secch_current = t_departure % self.section
                            t_secp_current = 0
                            t_secd_current = 0
                            if t_secch_current < self.min_rest:
                                r_charge = np.exp(5 * t_secch_current / 3600) - np.exp(3.75)
                            else:
                                # r_charge = -10 * (np.exp(1.5 * t_secch_current / 3600) - np.exp(1.125))
                                r_charge = -32 * t_secch_current / 3600 + 24
                        else: # still in current section
                            t_secch_current = t_stay + t_secch_current
                            if t_secch_current < self.min_rest:
                                r_charge = np.exp(5 * t_secch_current / 3600) - np.exp(3.75)
                            else:
                                # r_charge = -10 * (np.exp(1.5 * t_secch_current / 3600) - np.exp(1.125))
                                r_charge = -32 * t_secch_current / 3600 + 24
                    # if r_charge <= -175:
                    #     r_charge = -200
                else:
                    r_charge = 0
                    t_stay = 0
                    if t_arrival >= self.section:  # A new section begin before arrival next state
                        t_secp_current = 0
                        t_secch_current = 0

            # only the reward for a step, do not need to take totoal rest time into account
            r_parking = 0

        #next node is a parking lot
        else:
            if rest == 0:
                t_stay = 0
                r_parking = 0
                if t_arrival >= self.section:  # A new section begin before arrival next state
                    t_secp_current = 0
                    t_secch_current = 0
            else:
            # Calculate reward for suitable rest time in next node
                remain_rest = self.min_rest - t_secch_current - t_secp_current
                if remain_rest <= 0:# Get enough rest before arriving next parking loy
                    t_stay = 0
                    r_parking = -100
                    if t_arrival >= self.section:  # A new section begin before arrival next state
                        t_secp_current = 0
                        t_secch_current = 0
                else:
                    t_stay = rest * remain_rest
                    t_departure = t_arrival + t_stay
                    if t_arrival >= self.section:  # A new section begin before arrival next state
                        t_secp_current = t_stay
                        t_secch_current = 0
                        r_parking = -2 * (np.exp(5 * t_stay / 3600) - 1)
                    else:
                        if t_departure >= self.section:  # A new section begin before leaving next state
                            t_secp_current = t_departure % self.section
                            t_secch_current = 0
                            t_secd_current = 0
                            r_parking = -2 * (np.exp(5 * t_secp_current / 3600) - 1)
                        else:# still in current section
                            r_parking = -2 * (np.exp(5 * t_stay / 3600) - 1)
             # Reward for charging time for a step,
            if t_secch_current < self.min_rest:
                r_charge = np.exp(5 * t_secch_current / 3600) - np.exp(3.75)
            else:
                # r_charge = -10 * (np.exp(1.5 * t_secch_current / 3600) - np.exp(1.125))
                r_charge = -32 * t_secch_current / 3600 + 24

        # Calculate immediate reward
        r_distance_w = r_distance * 10
        r_energy_w = r_energy * 1500
        r_driving_w = r_driving * 1
        r_charge_w = r_charge * 0.1
        r_parking_w = r_parking * 1

        reward = r_distance_w + r_energy_w + r_charge_w + r_driving_w + r_parking_w
        print("r_distance, r_energy, r_charge, r_driving, r_parking_p = ", r_distance_w, r_energy_w, r_charge_w, r_driving_w, r_parking_w)
        print("reward = ", reward, "\n")


        # # update state
        #node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = self.state
        if next_node == 1:
            self.state = (1, nearest_x1, nearest_y1, charge, t_stay, t_secd_current, t_secp_current, t_secch_current)
        if next_node == 2:
            self.state = (2, nearest_x2, nearest_y2, charge, t_stay, t_secd_current, t_secp_current, t_secch_current)
        if next_node == 3:
            self.state = (3, nearest_x3, nearest_y3, charge, t_stay, t_secd_current, t_secp_current, t_secch_current)
        if next_node == 4:
            self.state = (4, nearest_x4, nearest_y4, soc_after_driving, t_stay, t_secd_current, t_secp_current, t_secch_current)
        if next_node == 5:
            self.state = (5, nearest_x5, nearest_y5, soc_after_driving, t_stay, t_secd_current, t_secp_current, t_secch_current)

        return np.array(self.state, dtype=np.float32), reward, terminated

    def reset(self):

        # s := (current_node, x1, y1, soc, t_stay, t_secd, t_secr, t_secch)
        node = random.choice([4, 5])
        data = pd.read_csv('parking_bbox.csv')
        # location = data.sample(n =1, random_state=42)
        location = data.sample(n=1)
        x = location['Latitude'].values[0]
        y = location['Longitude'].values[0]
        soc = random.uniform(0.1, 0.8)
        t_stay = 0
        t_secd = 0
        t_secr = 0
        t_secch = 0
        self.state = (node, x, y, soc, t_stay, t_secd, t_secr, t_secch)

        # if self.render_mode == "human":
        #     self.render()
        return np.array(self.state, dtype=np.float32), {}




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

