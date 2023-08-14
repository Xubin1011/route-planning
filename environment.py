# Date: 8/3/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...

from distance_haversine import haversine

import math
from typing import Optional, Tuple, Union

import numpy as np

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
# charge:= {0, 0.3, 0.5, 0.8}
# rest:= {0, 0.2, 0.4, 0.6, 0.8, 1}


## State/Observation Space is a np.ndarray, with shape `(5,)`, State s := (x1, y1, SoC, t_secd, t_secr, t_secch)
# | Num | State | Min | Max |
# | 1 | Distance from target | 0 | Distance from source | ##It is impossible for d to be greater than the maximum distance, no need to consider the range
# | 1 | x1,y1 current location| inf |
# | 2 | SoC | 0.1 | 0.8 |
# | 3 | t_secd | 0 | 4.5 |
# | 4 | t_ar | 0 | 5.25 |  ##t_ar will always be in this range, so there is no need to consider the range
# | 5 | t_de | 0 | 5.25 |  ##t_de will always be in this range, so there is no need to consider the range


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
# 3. Distance from target is 0, taht means target has benn reached


# human:The environment is continuously rendered in the current display or terminal
# rgb_array: Return a single frame representing the current state of the environment.
metadata = {
    "render_modes": ["human", "rgb_array"],
    "render_fps": 50,
}

class rp_env(gym.Env[np.ndarray, np.ndarray]):

    def __init__(self, render_mode: Optional[str] = None):
        #initialization
        self.x1 = 52.66181 #source
        self.y1 = 13.38251
        self.c1 = 47
        self.x2 = 51.772324 #target
        self.y2 = 12.402652
        self.c2 = 88
        self.m = 13500 #(Leergewicht)
        self.g = 9.81
        self.rho = 1.225
        self.A_front = 10.03
        self.c_r = 0.01
        self.c_d = 0.7
        self.a = 0

        # Aveage charge power
        self.average_charge_power = 100
        # Each section has the same fixed travel time
        self.min_rest = 0.75
        self.max_driving = 4.5
        self.section = self.min_rest + self.max_driving
        # Reward factor
        self.k1 = 1 # For the distance to the target
        self.k2 = 1 # For the battery’s operation limits
        self.k3 = 1 # For the suitable charging time
        self.k4 = 0.5 # For the suitable charging time, k4 < K3,
        # Angen tends to spend more time at charging stations than resting in parking lots




        # Fail the episode
        self.distance_min = 0
        self.distance_max = haversine(self.x1, self.y1, self.x2, self.y2)
        self.SoC_min = 0.1
        self.SoC_max = 0.8
        self.t_secd_min = 0
        self.t_secd_max = 4.5
        # self.t_ar = 5.25
        # self.t_de = 5.25
        
        high = np.array(
            [
                self.distance_max,
                np.finfo(np.float32).max,
                self.SoC_max,
                np.finfo(np.float32).max,
                self.t_secd_max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        
        low = np.array(
            [
                self.distance_min,
                np.finfo(np.float32).min,
                self.SoC_min,
                np.finfo(np.float32).min,
                self.t_secd_min,
                np.finfo(np.float32).min,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(24) ## 3*4+2*6
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None



    def reward_charge(self, soc, t_secr, t_secch, charge):
        # If next_node is 1, 2, 3, reward for charging time
        recharge = charge - soc
        if recharge > 0:
            charging_time = recharge/self.average_charge_power
            t_secch = t_secch + charging_time
            if t_secch < self.min_rest:
                r_charge = -k4 * (self.min_rest - t_secch)



    def step(self, action):
        #Run one timestep of the environment’s dynamics using the agent actions.
        #Calculate reward, update state
        #At the end of an episode, call reset() to reset this environment’s state for the next episode.
        
        
        #Check if the action is valid
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        #Obtain current state
        x1, y1, soc, t_secd, t_secr, t_secch = self.state
        print('x1, y1, SoC, t_secd, t_ar, t_de=', x1, y1, soc, t_secd, t_secr, t_secch) #test

        #Calculate reward, update state









        # Determine the type of render
        if self.render_mode == "human":
            self.render()
        # return new state and indicate whether to stop the episode
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

