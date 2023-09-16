# Date: 8/3/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...
import random
import sys

from nearest_location import nearest_location
from consumption_duration import consumption_duration
from consumption_duration import haversine
from way_calculation import way

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
# node_next:= {1,2,3,4,5}
# charge:= {0, 0.3, 0.5, 0.8}
# rest:= {0, 0.3, 0.6, 0.9, 1}


## State/Observation Space is a np.ndarray, with shape `(8,)`,
# State s := (current_node, x1, y1, soc,t_stay, t_secd, t_secr, t_secch)


# Rewards
# r1: Reward for the distance to the target
# r2: Reward based on Battery’s operation limits
# r3: Reward for the suitable charging time
# r4: Reward for the suitable driving time
# r5: Reward for the suitable rest time at parking lots


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

        self.w_distance = 1000  #value range -1~+1
        self.w_energy = 1500 # -6~0.4
        self.w_driving = 1  #-100~0
        self.w_charge = 0.1 # -232~0
        self.w_parking = 10 # -100~0
        self.w_target = 0 # 1 or -1

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
        node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = self.state
        # print(node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current) #test

        # Obtain selected action
        index_cpu = action.cpu()
        node_next, charge, rest = self.df_actions.iloc[index_cpu.item()]
        print('node_next, charge, rest = ', node_next, charge, rest)

        next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters = self.myway.info_way(node_current, x_current, y_current, node_next)

        print("next_x, next_y, d_next, power_next, consumption, typical_duration=", next_x, next_y, d_next, power_next, consumption, typical_duration)
        print("Length, average speed, average consumption", length_meters / 1000, "km", length_meters / typical_duration * 3.6, "km/h", consumption / length_meters * 100000, "kWh/100km\n")

        # the distance from current location to target
        d_current = haversine(x_current, y_current, self.myway.x_target, self.myway.y_target)
        # soc after driving
        soc_after_driving = soc - consumption / self.battery_capacity
        # the time that arriving next location
        t_arrival = t_secd_current + t_secch_current + t_secp_current + typical_duration
        # the driving time when arrive next location
        t_secd_current = t_secd_current + typical_duration

        # Calculate reward for distance
        r_distance = (d_current - d_next) / 25000
        if d_next == 0:
            # r_distance = self.target
            terminated = True
            print("Terminated: Arrival target")
        # else:
        #     # r_distance = np.exp * ((d_current - d_next) / 25000) - 1
        #     r_distance = (d_current - d_next) / 25000

        # Reward for battery
        # If there is recuperated energy, the soc can be charged up to 0.8
        if consumption < 0:
            if soc_after_driving > 0.8:
                soc_after_driving = 0.8

        # Punishment for the trapped on the road
        if soc_after_driving < 0:  # Trapped
            terminated = True
            r_energy = - 6
            print("Terminated: Trapped on the road, should be reseted")
        else:  # No trapped
            if soc_after_driving < 0.1:  # Still can run, but violated constraint
                # r_energy =  np.log(self.w3 * abs(soc_after_driving - 0.1))
                r_energy = np.log(0.1 * soc_after_driving) + 5
                self.num_trapped += 1
                if self.num_trapped == 10:
                    terminated = True  # Violate the self.max_trapped times, stop current episode
                    print("Terminated: Violated soc 10 times,should be reseted")
            else:
                r_energy = 0.4  # No trapped

        # Calculate reward for suitable driving time when arriving next node
        # update t_secd_current
        if t_arrival >= self.section:  # A new section begin before arrival next state, only consider the reward of last section
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
                r_driving = np.exp((self.section - rest_time) / 3600) - np.exp(4.5)
        else:  # still in current section when arriving next poi
            if t_secd_current <= self.max_driving:
                r_driving = np.exp(t_secd_current / 3600) - np.exp(4.5)
            else:
                print("Terminated: Violated self.max_driving times,should be reseted")
                terminated = True
                # r_driving = -10 * (t_secd_current- self.max_driving)
                r_driving = -100

        # next node is an charging station
        # update t_stay, t_secch_current,t_secp_current
        if node_next in range(self.myway.n_ch):
            # not a parking lot, only take total rest time into account
            r_parking = -2 * (np.exp(5 * t_secp_current / 3600) - 1)
            # Calculate reward for suitable charging time in next node
            if charge > soc_after_driving:  # must be charged at next node
                t_stay = (charge - soc_after_driving) * self.battery_capacity / power_next * 3600  # in s
                t_departure = t_arrival + t_stay
                if t_arrival >= self.section:  # A new section begin before arrival next state,only consider the reward of last section
                    if t_secch_current < self.min_rest:
                        r_charge = np.exp(5 * t_secch_current / 3600) - np.exp(3.75)
                    else:
                        # r_charge = -10 * (np.exp(1.5 * t_secch_current / 3600) - np.exp(1.125))
                        r_charge = -32 * t_secch_current / 3600 + 24
                    t_secp_current = 0
                    t_secch_current = t_stay
                else:
                    if t_departure >= self.section:  # A new section begin before leaving next state,only consider the reward of last section
                        t_secch_current += (t_stay - t_departure % self.section)
                        if (t_stay - t_departure % self.section) < 0:
                            print("Warning! wrong Value of t_secch_current")
                            sys.exit(1)
                        if t_secch_current < self.min_rest:
                            r_charge = np.exp(5 * t_secch_current / 3600) - np.exp(3.75)
                        else:
                            # r_charge = -10 * (np.exp(1.5 * t_secch_current / 3600) - np.exp(1.125))
                            r_charge = -32 * t_secch_current / 3600 + 24
                        t_secch_current = t_departure % self.section
                        t_secp_current = 0
                        t_secd_current = 0
                    else:  # still in current section
                        t_secch_current = t_stay + t_secch_current
                        if t_secch_current < self.min_rest:
                            r_charge = np.exp(5 * t_secch_current / 3600) - np.exp(3.75)
                        else:
                            # r_charge = -10 * (np.exp(1.5 * t_secch_current / 3600) - np.exp(1.125))
                            r_charge = -32 * t_secch_current / 3600 + 24
                # if r_charge <= -175:
                #     r_charge = -200
            else: # No need to charge
                charge = soc_after_driving
                t_stay = 0
                if t_secch_current < self.min_rest: # A new section begins before arrival next state or still in current section
                    r_charge = np.exp(5 * t_secch_current / 3600) - np.exp(3.75)
                else:
                    r_charge = -32 * t_secch_current / 3600 + 24

                if t_arrival >= self.section:  # A new section begins before arrival next state
                    t_secp_current = 0
                    t_secch_current = 0

        # next node is a parking lot
        else:
            # no charge at a parking lot, only take total charge time into account
            if t_secch_current < self.min_rest:  # A new section begins before arrival or departure next state or still in current section
                r_charge = np.exp(5 * t_secch_current / 3600) - np.exp(3.75)
            else:
                r_charge = -32 * t_secch_current / 3600 + 24

            # Calculate reward for suitable rest time in next node
            remain_rest = self.min_rest - t_secch_current - t_secp_current
            t_stay = remain_rest * rest
            if t_stay <= 0:  # Get enough rest before arriving next parking loy
                t_stay = 0
                r_parking = -100
                if t_arrival >= self.section:  # A new section begin before arrival next state
                    t_secp_current = 0
                    t_secch_current = 0
            else:# t_stay > 0
                t_departure = t_arrival + t_stay
                if t_arrival >= self.section:  # A new section begin before arrival next state
                    r_parking = -2 * (np.exp(5 * t_secp_current / 3600) - 1) # the reward of last section
                    t_secp_current = t_stay
                    t_secch_current = 0
                else:
                    if t_departure >= self.section:  # A new section begin before leaving next state
                        t_secp_current += (t_stay - t_departure % self.section)
                        if (t_stay - t_departure % self.section) < 0:
                            print("Warning! wrong Value of t_secch_current")
                            sys.exit(1)
                        r_parking = -2 * (np.exp(5 * t_secp_current / 3600) - 1)# the reward of last section
                        t_secp_current = t_departure % self.section
                        t_secch_current = 0
                        t_secd_current = 0
                    else:  # still in current section
                        t_secp_current += t_stay
                        r_parking = -2 * (np.exp(5 * t_secp_current / 3600) - 1)

        if terminated == True and d_next == 0: # arrival target
            r_end = 1
        else:
            r_end = - 1

        # Calculate immediate reward
        r_distance_w = r_distance * self.w_distance
        r_energy_w = r_energy * self.w_energy
        r_driving_w = r_driving * self.w_driving
        r_charge_w = r_charge * self.w_charge
        r_parking_w = r_parking * self.w_parking
        r_terminated_w = r_end * self.w_target

        reward = r_distance_w + r_energy_w + r_charge_w + r_driving_w + r_parking_w + r_terminated_w
        print("r_distance, r_energy, r_charge, r_driving, r_parking_p, r_end = ", r_distance_w, r_energy_w, r_charge_w,
              r_driving_w, r_parking_w, r_terminated_w)
        print("reward = ", reward, "\n")

        # # update state
        # node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = self.state
        # next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters = self.myway.info_way(
            # node_current, x_current, y_current, node_next)
        if node_next in range(self.myway.n_ch):
            self.state = (node_next, next_x, next_y, charge, t_stay, t_secd_current, t_secp_current, t_secch_current)
        else:
            self.state = (node_next, next_x, next_y, soc_after_driving, t_stay, t_secd_current, t_secp_current, t_secch_current)
        
        return np.array(self.state, dtype=np.float32), reward, terminated

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

























