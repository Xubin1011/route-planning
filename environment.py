# Date: 8/3/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...
import random

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



# human:The environment is continuously rendered in the current display or terminal
# rgb_array: Return a single frame representing the current state of the environment.
metadata = {
    "render_modes": ["human", "rgb_array"],
    "render_fps": 50,
}

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
        # Reward factor
        self.w1 = 1 # For the distance to the target
        self.w2 = 5 # Punishment for the trapped on the road
        self.w3 = 0.1 # Still can run, but violated constraint
        self.w4 = 0.4 # No trapped
        self.w5 = 1 # For the suitable charging time
        self.w6 = 0.05 # For the suitable charging time, w6 < w5,
        # Angen tends to spend more time at charging stations than resting in parking lots
        self.w7 = 10  # Reward for the suitable rest time at parking lots
        self.w8 = 1 ## Must rest at parking lots
        self.w9 = 1000 # Exceeded the max. driving time in a section
        self.w10 = 1 # For the suitable driving time

        self.w_distance = 1
        self.w_trapped = 1
        self.w_charge = 1
        self.w_rest = 1
        self.w_driving = 1

        self.num_trapped = 0 # The number that trapped on the road
        self.max_trapped = 10

        # # Initialize the actoin space
        # self.next_node = np.array([1, 2, 3, 4, 5])
        # self.charge = np.array([0, 0.3, 0.5, 0.8])
        # self.rest = np.array([0, 0.3, 0.6, 0.9, 1])
        # next_node_space = spaces.Discrete(len(self.next_node))
        # charge_space = spaces.Discrete(len(self.charge))
        # rest_space = spaces.Discrete(len(self.rest))
        # self.action_space = spaces.Tuple((next_node_space, charge_space, rest_space))
        self.action_space = spaces.Discrete(22)
        self.df_actions = pd.read_csv("actions.csv")


        self.state = None

        # #Initialize the render mode
        # self.render_mode = render_mode
        #
        # self.screen_width = 600
        # self.screen_height = 400
        # self.screen = None
        # self.clock = None
        # self.isopen = True

    # def action_space_sample(self):
    #     random_next_node = np.random.choice(self.next_node)
    #     if random_next_node in [1, 2, 3]:
    #         random_charge = np.random.choice(self.charge)
    #         action = (random_next_node, random_charge, 0)
    #         print("action:", action)
    #     else:
    #         random_rest = np.random.choice(self.rest)
    #         action = (random_next_node, 0, random_rest)
    #         print("action:", action)
    #     return(action)

    # def action_space_sample(self):
    #     df = pd.read_csv("actions.csv")
    #     random_index = random.randint(0, len(df) - 1)
    #     random_action = df.iloc[random_index]
    #     return(random_action)


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
        # If current POI is a parking lot, t_secr_current includes rest  time at the current location
        node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secr_current, t_secch_current = self.state
        # print(node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secr_current, t_secch_current) #test

        #Obtain selected action
        index_cpu = action.cpu()
        next_node, charge, rest = self.df_actions.iloc[index_cpu.item()]
        print('next_node, charge, rest = ', next_node, charge, rest)

        # Obtain the altitude and/or power of current location
        if node_current in [1, 2, 3]:  # charging station
            c_current, power_current = self.cs_elevation_power(x_current, y_current)
        else:  # parking lots
            c_current = self.p_elevation(x_current, y_current)

        # # Obtain the altitude and/or power of next location
        # if next_node in [1, 2, 3]:  # charging station
        #     c_next, power_next = self.cs_elevation_power(x_next, y_next)
        # else:  # parking lots
        #     c_next = self.p_elevation(x_next, y_next)


        # Determine whether it is in a new section
        # updata t_secd_current, t_secr_current, t_secch_current at current location
        t_departure = t_secd_current + t_secr_current + t_secch_current
        t_arrival = t_departure - t_stay
        if t_arrival >= self.section:  # new section begin before arriving current location
            t_secd_current = t_arrival % self.section
            t_secr_current = 0
            t_secch_current = 0
        else:  # still in current section
            if t_departure >= self.section:
                t_secd_current = 0
                if node_current in [1, 2, 3]:
                    t_secch_current = t_departure - self.section
                    t_secr_current = 0
                else:
                    t_secch_current = 0
                    t_secr_current = t_departure - self.section


        # Obtain n nearest POIs
        nearest_ch = nearest_location(self.file_path_ch, x_current, y_current, self.n_ch)
        print(nearest_ch)
        nearest_x1 = nearest_ch.loc[0, 'Latitude']
        nearest_y1 = nearest_ch.loc[0, 'Longitude']
        nearest_x2 = nearest_ch.loc[1, 'Latitude']
        nearest_y2 = nearest_ch.loc[1, 'Longitude']
        nearest_x3 = nearest_ch.loc[2, 'Latitude']
        nearest_y3 = nearest_ch.loc[2, 'Longitude']
        # print('nearest_1-3:', nearest_x1, nearest_y1, nearest_x2, nearest_y2, nearest_x3, nearest_y3)

        nearest_p = nearest_location(self.file_path_p, x_current, y_current, self.n_p)
        print(nearest_p)
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
        print("Length, speed, consumption", length_meters/1000, "m", length_meters/typical_duration * 3.6, "km/h", consumption/length_meters*100000, "kWh/100km\n")
        #Calculate reward for distance
        d_current = haversine(x_current, y_current, self.x_target, self.y_target)
        if d_next == 0:
            r_distance = 100
            terminated = True
        else:
            r_distance = self.w1 * (d_current - d_next)
            
        # Reward for battery
        # soc after driving
        soc_after_driving = soc - consumption / self.battery_capacity
        # If there is recuperated energy, the soc can be charged up to 0.8
        if consumption < 0:
            if soc_after_driving > 0.8:
                soc_after_driving = 0.8

        # Punishment for the trapped on the road
        if soc_after_driving < 0: # Trapped
            terminated = True
            r_trapped = - self.w2
            print("trapped on the road, should be reseted")
        else: # No trapped
            if soc_after_driving < 0.1: # Still can run, but violated constraint
                #r_trapped =  math.log(self.w3 * abs(soc_after_driving - 0.1))
                r_trapped = math.log(self.w3 * soc_after_driving) + 5
                self.num_trapped = self.num_trapped + 1
                if self.num_trapped == self.max_trapped:
                    terminated = True # Violate the self.max_trapped times, stop current episode

                    print("Violated self.max_trapped times,should be reseted")
            else:
                r_trapped = self.w4 # No trapped



        if next_node in [1, 2, 3]: # Calculate reward for suitable charging time in next node
            if charge >= soc_after_driving:
                t_charge_next = (charge - soc_after_driving) / next_power
                t_secch_current = t_secch_current + t_charge_next
                if t_secch_current <= self.min_rest:
                    r_charge = -self.w5 * (self.min_rest - t_secch_current)
                else:
                    r_charge = self.w6 * (self.min_rest - t_secch_current)
            else:
                r_charge = -10
                t_charge_next = 0


        else: # Calculate reward for suitable rest time in next node
            remain_rest = self.min_rest - t_secch_current
            if remain_rest < 0:# Get enough rest at charging stations
                if rest != 0:
                    r_rest = self.w7 * rest
                    t_rest_next = 0  # Action must be modified

            else:
                t_rest_next = rest * remain_rest
                t_secr_current = t_secr_current + t_rest_next # Must rest at parking lots
                r_rest = - self.w8 * t_secr_current

        # Calculate reward for suitable driving time before leaving next node
        t_secd_current = t_secd_current + typical_duration
        if t_secd_current > self.max_driving:
            terminated = True
            r_driving = -self.w9 * (t_secd_current - self.max_driving)
        else:
            t_tem = self.max_driving - t_secd_current
            r_driving = - self.w10 * t_tem

        # Calculate immediate reward
        if next_node in [1, 2, 3]:
            reward = self.w_distance * r_distance + self.w_trapped * r_trapped + self.w_charge * r_charge + self.w_driving * r_driving
            print("r_distance, r_trapped, r_charge, r_driving = ", r_distance, r_trapped, r_charge, r_driving)
            print("reward = ", reward, "\n")
        else:
            reward = self.w_distance * r_distance + self.w_trapped * r_trapped + self.w_rest *  r_rest + self.w_driving * r_driving
            print("r_distance, r_trapped, r_rest, r_driving = ", r_distance, r_trapped, r_rest, r_driving)
            print("reward = ", reward, "\n")


        # # update state
        #node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secr_current, t_secch_current = self.state
        if next_node == 1:
            self.state = (1, nearest_x1, nearest_y1, charge, t_charge_next, t_secd_current, t_secr_current, t_secch_current)
        if next_node == 2:
            self.state = (2, nearest_x2, nearest_y2, charge, t_charge_next, t_secd_current, t_secr_current, t_secch_current)
        if next_node == 3:
            self.state = (3, nearest_x3, nearest_y3, charge, t_charge_next, t_secd_current, t_secr_current, t_secch_current)
        if next_node == 4:
            self.state = (4, nearest_x4, nearest_y4, soc_after_driving, t_rest_next, t_secd_current, t_secr_current, t_secch_current)
        if next_node == 5:
            self.state = (5, nearest_x5, nearest_y5, soc_after_driving, t_rest_next, t_secd_current, t_secr_current, t_secch_current)


        # # Determine the type of render
        # if self.render_mode == "human":
        #     self.render()
        # return new state and indicate whether to stop the episode
        return np.array(self.state, dtype=np.float32), reward, terminated

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # maybe_parse_reset_bounds can be called during a reset() to customize the sampling
        # ranges for setting the initial state distributions.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high

        # s := (current_node, x1, y1, soc, t_stay, t_secd, t_secr, t_secch)
        node = random.choice([4, 5])
        data = pd.read_csv('parking_bbox.csv')
        location = data.sample(n =1, random_state=42)
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




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

