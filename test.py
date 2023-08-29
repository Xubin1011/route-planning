import pandas as pd

import random

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
from environment import rp_env

# env = rp_env()
#
# sampled_action = env.action_space.sample()
# print("Sampled Action:", sampled_action)



#
# state, info = env.reset()
# n_observations = len(state)
#
#
# print(n_observations)


import numpy as np


# # Initialize the action space
# next_node = np.array([1, 2, 3, 4, 5])
# charge = np.array([0, 0.3, 0.5, 0.8])
# rest = np.array([0, 0.3, 0.6, 0.9, 1])
# next_node_space = spaces.Discrete(len(next_node))
# charge_space = spaces.Discrete(len(charge))
# rest_space = spaces.Discrete(len(rest))
# action_space = spaces.Tuple((next_node_space, charge_space, rest_space))

# Create a sample environment
env = rp_env()

n_actions = env.action_space.n
print(n_actions)

# Sample a random action
sampled_action = env.action_space.sample()
print("Sampled Action:", sampled_action)

