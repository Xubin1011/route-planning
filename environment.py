# Date: 8/3/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...

from distance_haversine import haversine
from nearest_location import nearest_location
from consumption_duration import consumption_duration

import math
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.vector import VectorEnv
from gymnasium.vector.utils import batch_space





# The environment of route planning,
#  An electric vehicles need to choose the next action according to the current state,
#  the goal is to obtain the shortest time route


## Action Space is a 'ndarry', with shape `(3,)`, Action a:=(next_node, charge, rest )
# next_node:= {1,2,3,4,5}
# charge:= {0.3, 0.5, 0.8}
# rest:= {0, 0.3, 0.6, 0.9, 1}


## State/Observation Space is a np.ndarray, with shape `(7,)`,
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
# 1. SoC is less than 0.1 or greater than 0.8, which violates the energy constraint 
# 2. t_secd is greater than 4.5, which violates the time constraint
# 2. Episode length is greater than 500



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
        self.m = 13500 #(Leergewicht)
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


        # Aveage charge power
        self.average_charge_power = 100
        # Limitation of battery
        self.soc_min = 0.1
        self.soc_max = 0.8
        # Each section has the same fixed travel time
        self.min_rest = 0.75
        self.max_driving = 4.5
        self.section = self.min_rest + self.max_driving
        # Reward factor
        self.k1 = 1 # For the distance to the target
        self.k2 = 1000 # Punishment for the trapped on the road
        self.k3 = 1 # For the suitable charging time
        self.k4 = 0.5 # For the suitable charging time, k4 < K3,
        # Angen tends to spend more time at charging stations than resting in parking lots
        self.k5 = 1 # For the suitable driving time
        self.k6 = 100 # Penalties for violating driving time constraints
        self.k7 = 1 # Reward for the suitable rest time at parking lots

        #Initialize the actoin space
        next_node = np.array([1, 2, 3, 4, 5])
        charge = np.array([0, 0.3, 0.5, 0.8])
        rest = np.array([0, 0.3, 0.6, 0.9, 1])
        next_node_space = spaces.Discrete(len(next_node))
        charge_space = spaces.Discrete(len(charge))
        rest_space = spaces.Discrete(len(rest))
        self.action_space = spaces.Tuple((next_node_space, charge_space, rest_space))

        #Initialize the render mode
        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    def cs_elevation_power(self,x1, y1):
        matching_row = self.data_ch[(self.data_ch["Latitude"] == x1) & (self.data_ch["Longitude"] == y1)]
        if not matching_row.empty:
            cs_elevation = matching_row["Elevation"].values[0]
            cs_power = matching_row["Power"].values[0]
        else:
            print("Current location not found in the dataset of cd")

        return(cs_elevation, cs_power)

    def p_elevation(self,x1, y1):
        matching_row = self.data_p[(self.data_p["Latitude"] == x1) & (self.data_p["Longitude"] == y1)]
        if not matching_row.empty:
            p_elevation = matching_row["Altitude"].values[0]
        else:
            print("Current location not found in the dataset of p")

        return(p_elevation)

    def reward_charge(self, soc, t_secch, charge):
        # If next_node is 1, 2, 3, calculate the reward for charging time
        recharge = charge - soc
        if recharge >= 0:
            charging_time = recharge/self.average_charge_power
            t_secch = t_secch + charging_time
            if t_secch <= self.min_rest:
                r_charge = -self.k3 * (self.min_rest - t_secch)
                terminated = 1
            else:
                r_charge = self.k4 * (self.min_rest - t_secch)
                terminated = 1
        else:
            r_charge = 0
            terminated = 1
        return (r_charge, terminated)

    def step(self, action):
        #Run one timestep of the environment’s dynamics using the agent actions.
        #Calculate reward, update state
        #At the end of an episode, call reset() to reset this environment’s state for the next episode.
        
        
        #Check if the action is valid
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        #Obtain current state
        # If current POI is a charging station, soc is battery capacity that after charging, t_secch_current includes charging time at the current location
        # If current POI is a parking lot, t_secr_current includes rest  time at the current location
        node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secr_current, t_secch_current = self.state
        print(node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secr_current, t_secch_current) #test

        #Obtain selected action
        next_node, charge, rest = action
        print('next_node, charge, rest', next_node, charge, rest)

        # Obtain the altitude and/or power of current location
        if node_current in [1, 2, 3]:  # charging station
            c_current, power_current = self.cs_elevation_power(x_current, y_current)
        else:  # parking lots
            c_current = self.p_elevation(x_current, y_current)


        # Determine whether it is in a new section
        # updata t_secd_current, t_secr_current, t_secch_current at current location
        t_departure = t_secd_current + t_secr_current + t_secch_current
        t_arrival = t_departure - t_stay
        if t_arrival >= self.section: # new section begin before arriving current location
            t_secd_current = t_arrival % self.section
            t_secr_current = 0
            t_secch_current = 0
        else: #still in current section
            if t_departure >= self.section:
                t_secd_current = 0
                if node_current in [1, 2, 3]:
                    t_secch_current = t_departure - self.section
                    t_secr_current = 0
                else:
                    t_secch_current = 0
                    t_secr_current = t_departure - self.section




        # Stop criterion  ????????????
        #?????????????????????????????????








        # Obtain n nearest POIs
        nearest_ch = nearest_location(self.file_path_ch, x_current, y_current, self.n_ch)
        print(nearest_ch)
        nearest_x1 = nearest_ch.loc[0, 'Latitude']
        nearest_y1 = nearest_ch.loc[0, 'Longitude']
        nearest_x2 = nearest_ch.loc[1, 'Latitude']
        nearest_y2 = nearest_ch.loc[1, 'Longitude']
        nearest_x3 = nearest_ch.loc[2, 'Latitude']
        nearest_y3 = nearest_ch.loc[3, 'Longitude']
        print('nearest_1-3:', nearest_x1, nearest_y1, nearest_x2, nearest_y2, nearest_x3, nearest_y3)

        nearest_p = nearest_location(self.file_path_p, x_current, y_current, self.n_p)
        print(nearest_p)
        nearest_x4 = nearest_p.loc[0, 'Latitude']
        nearest_y4 = nearest_p.loc[0, 'Longitude']
        nearest_x5 = nearest_p.loc[1, 'Latitude']
        nearest_y5 = nearest_p.loc[1, 'Longitude']
        print('nearest_4-5:', nearest_x4, nearest_y4, nearest_x5, nearest_y5)





        if next_node == 1:
            d_next = haversine(nearest_x1, nearest_y1, self.x_target, self.y_target)
            #consumption and typical_duration from current location to next node
            nearest_c1 = self.cs_elevation_power(nearest_x1, nearest_y1)
            consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, c_current, nearest_x1, nearest_y1,
                                                                                nearest_c1, self.m, self.g,
                                                                                self.c_r, self.rho, self.A_front,
                                                                                self.c_d, self.a, self.eta_m,
                                                                                self.eta_battery)
        if next_node == 2:
            d_next = haversine(nearest_x2, nearest_y2, self.x_target, self.y_target)
            nearest_c2 = self.cs_elevation_power(nearest_x2, nearest_y2)
            consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, c_current,
                                                                                nearest_x2, nearest_y2,
                                                                                nearest_c2, self.m, self.g,
                                                                                self.c_r, self.rho, self.A_front,
                                                                                self.c_d, self.a, self.eta_m,
                                                                                self.eta_battery)
        if next_node == 3:
            d_next = haversine(nearest_x3, nearest_y3, self.x_target, self.y_target)
            nearest_c3 = self.cs_elevation_power(nearest_x3, nearest_y3)
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

        #Calculate reward for distance
        d_current = haversine(x_current, y_current, self.x_target, self.y_target)
        r_distance = self.k1 * (d_current - d_next)

        #Punishment for the trapped on the road ???????????????????

        # if charge-consumption < 0.1:   ????????????????????????
        #     terminated = 1             ???????????????????
        #     return (0, terminated)
















        # Determine the type of render
        if self.render_mode == "human":
            self.render()
        # return new state and indicate whether to stop the episode
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

