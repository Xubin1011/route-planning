# Date: 8/3/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...
import random
import sys

from nearest_location import nearest_location
from consumption_duration import consumption_duration
from consumption_duration import haversine
from way_noloops import way
from global_var import initial_data_p, initial_data_ch, data_p, data_ch

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


## Action Space is a tuple, with 3 arrays, Action a:=(node_next, charge, rest )
# node_next:= {1,2,3,4,5} or more
# charge:= {0, 0.3, 0.5, 0.8}
# rest:= {0, 0.3, 0.6, 0.9, 1}


## State/Observation Space is a np.ndarray, with shape `(8,)`,
# State s := (current_node, x1, y1, soc,t_stay, t_secd, t_secr, t_secch)

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
        # initialization
        # Limitation of battery
        self.battery_capacity = 588  # (in kWh)
        self.soc_min = 0.1
        self.soc_max = 0.8
        # Each section has the same fixed travel time
        self.min_rest = 2700  # in s
        self.max_driving = 16200  # in s
        self.section = self.min_rest + self.max_driving

        self.num_trapped = 0  # The number that trapped on the road
        self.max_trapped = 10

        # # Initialize the actoin space, state space
        self.df_actions = pd.read_csv("actions.csv")
        self.action_space = spaces.Discrete(self.df_actions.shape[0])
        self.state = None
        
        self.myway = way()

    def step(self, action):
        # Run one timestep of the environment’s dynamics using the agent actions.
        # Calculate reward, update state
        # At the end of an episode, call reset() to reset this environment’s state for the next episode.

        terminated = False
        # Check if the action is valid
        # assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        # Obtain current state
        # If current POI is a charging station, soc is battery capacity that after charging, t_secch_current includes charging time at the current location
        # If current POI is a parking lot, t_secp_current includes rest  time at the current location
        node_current, index_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = self.state
        # print(node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current) #test
        x_current, y_current, alti_current, power = self.myway.geo_coord(node_current, index_current)

        # Obtain selected action
        # index_cpu = action.cpu()
        # node_next, charge, rest = self.df_actions.iloc[index_cpu.item()]
        node_next, charge, rest = self.df_actions.iloc[action]
        print('node_next, charge, rest = ', node_next, charge, rest)

        index_next, next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters = self.myway.info_way(node_current, x_current, y_current, alti_current, node_next)

        print("next_x, next_y, d_next, power_next, consumption, typical_duration=", next_x, next_y, d_next, power_next, consumption, typical_duration)
        print("Length, average speed, average consumption", length_meters / 1000, "km", length_meters / typical_duration * 3.6, "km/h", consumption / length_meters * 100000, "kWh/100km\n")
    ##################################################################
        # the distance from current location to target
        d_current = haversine(x_current, y_current, self.myway.x_target, self.myway.y_target)
        # soc after driving
        soc_after_driving = soc - consumption / self.battery_capacity
        # the time that arriving next location
        t_arrival = t_secd_current + t_secch_current + t_secp_current + typical_duration
        # the driving time when arrive next location
        t_secd_current = t_secd_current + typical_duration
        ##################################################################
        # check target
        if d_next <= 25000 and soc_after_driving >= 0.1:
            # r_distance = self.target
            terminated = True
            print("Terminated: Arrival target")
        ###################################################################
        # check soc constraint
        if soc_after_driving < 0:  # Trapped
            terminated = True
            print("Terminated: Trapped on the road, should be reseted")
        else:  # No trapped
            if soc_after_driving < 0.1:  # Still can run, but violated constraint
                self.num_trapped += 1
                if self.num_trapped == self.max_trapped:
                    terminated = True  # Violate the self.max_trapped times, stop current episode
                    print(f"Terminated: Violated soc {self.max_trapped} times,should be reseted")
            else:
                terminated = False  # No trapped
        ##################################################################
        # check rest, driving time constraint
        if t_arrival >= self.section:  # A new section begin before arrival next state, only consider the reward of last section
            t_secd_current = t_arrival % self.section
            rest_time = t_secp_current + t_secch_current
            if rest_time < self.min_rest:
                terminated = True
                print("Terminated: Violated self.max_driving times,should be reseted")
        else:  # still in current section when arriving next poi
            if t_secd_current >= self.max_driving:
                print("Terminated: Violated self.max_driving times,should be reseted")
                terminated = True
        ##################################################################
        # next node is a charging station
        # update t_stay, t_secch_current,t_secp_current
        if node_next in range(self.myway.n_ch):
            if charge > soc_after_driving:  # must be charged at next node
                t_stay = (charge - soc_after_driving) * self.battery_capacity / power_next * 3600  # in s
                t_departure = t_arrival + t_stay
                if t_arrival >= self.section:  # A new section begin before arrival next state,only consider the reward of last section
                    t_secp_current = 0
                    t_secch_current = t_stay
                else:
                    if t_departure >= self.section:  # A new section begin before leaving next state,only consider the reward of last section
                        t_secch_current += (t_stay - t_departure % self.section)
                        if (t_stay - t_departure % self.section) < 0:
                            print("Warning! wrong Value of t_secch_current")
                            sys.exit(1)
                        t_secch_current = t_departure % self.section
                        t_secp_current = 0
                        t_secd_current = 0
                    else:  # still in current section
                        t_secch_current = t_stay + t_secch_current
            else: # No need to charge
                charge = soc_after_driving
                t_stay = 0
                if t_arrival >= self.section:  # A new section begins before arrival next state
                    t_secp_current = 0
                    t_secch_current = 0
        ##################################################################
        # next node is a parking lot
        else:
            # Calculate reward for suitable rest time in next node
            remain_rest = self.min_rest - t_secch_current - t_secp_current
            t_stay = remain_rest * rest
            if t_stay <= 0:  # Get enough rest before arriving next parking loy
                t_stay = 0
                if t_arrival >= self.section:  # A new section begin before arrival next state
                    t_secp_current = 0
                    t_secch_current = 0
            else:# t_stay > 0
                t_departure = t_arrival + t_stay
                if t_arrival >= self.section:  # A new section begin before arrival next state
                    t_secp_current = t_stay
                    t_secch_current = 0
                else:
                    if t_departure >= self.section:  # A new section begin before leaving next state
                        t_secp_current += (t_stay - t_departure % self.section)
                        if (t_stay - t_departure % self.section) < 0:
                            print("Warning! wrong Value of t_secch_current")
                            sys.exit(1)
                        t_secp_current = t_departure % self.section
                        t_secch_current = 0
                        t_secd_current = 0
                    else:  # still in current section
                        t_secp_current += t_stay
        ##################################################################
        # # update state
        # node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = self.state
        # next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters = self.myway.info_way(
            # node_current, x_current, y_current, node_next)
        if node_next in range(self.myway.n_ch):
            self.state = (node_next, next_x, next_y, charge, t_stay, t_secd_current, t_secp_current, t_secch_current)
        else:
            self.state = (node_next, next_x, next_y, soc_after_driving, t_stay, t_secd_current, t_secp_current, t_secch_current)
        
        return np.array(self.state, dtype=np.float32), terminated, d_next

    def reset(self):

        # s := (current_node, x1, y1, soc, t_stay, t_secd, t_secr, t_secch)
        node = random.randint(6, 9)
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

























