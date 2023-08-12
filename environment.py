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
# charge:= {0, 0.5, 0.8, 1}
# rest:= {0, 0.2, 0.4, 0.6, 0.8, 1}


## State/Observation Space is a np.ndarray, with shape `(5,)`, State s := (d, SoC, t_secd, t_ar, t_de)
# | Num | State | Min | Max |
# | 1 | Distance from target | 0 | Distance from source |
# | 2 | SoC | 0.1 | 0.8 |
# | 3 | t_secd | 0 | 4.5 |
# | 4 | t_ar | 0 | 5.25 |
# | 5 | t_de | 0 | 5.25 |


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
# 1. one of states is outside the allowed range
# 2. Episode length is greater than ?


metadata = {
    "render_modes": ["human", "rgb_array"],
    "render_fps": 50,
}

class rp_env(gym.Env[np.ndarray, np.ndarray]):

    def __init__(self, render_mode: Optional[str] = None):
        #initialization
        self.x1 = 52.66181
        self.y1 = 13.38251
        self.c1 = 47
        self.x2 = 51.772324
        self.y2 = 12.402652
        self.c2 = 88
        self.m = 13500 #(Leergewicht)
        self.g = 9.81
        self.rho = 1.225
        self.A_front = 10.03
        self.c_r = 0.01
        self.c_d = 0.7
        self.a = 0

        # Fail the episode
        self.distance_threshold = haversine(self.x1, self.y1, self.x2, self.y2)
        self.SoC = 0.7

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

