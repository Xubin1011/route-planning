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

env = rp_env()
#
# sampled_action = env.action_space.sample()
# print("Sampled Action:", sampled_action)



# # Initialize the action space
# next_node = np.array([1, 2, 3, 4, 5])
# charge = np.array([0, 0.3, 0.5, 0.8])
# rest = np.array([0, 0.3, 0.6, 0.9, 1])
# next_node_space = spaces.Discrete(len(next_node))
# charge_space = spaces.Discrete(len(charge))
# rest_space = spaces.Discrete(len(rest))
# action_space = spaces.Tuple((next_node_space, charge_space, rest_space))

# Create a sample environment

# Sample a random action
# sampled_action = env.action_space.sample()
# print("Sampled Action:", sampled_action)

from gymnasium.spaces import MultiDiscrete
import numpy as np
next_node = np.array([1, 2, 3, 4, 5])
charge = np.array([0, 0.3, 0.5, 0.8])
rest = np.array([0, 0.3, 0.6, 0.9, 1])
next_node_space = spaces.Discrete(len(next_node))
charge_space = spaces.Discrete(len(charge))
rest_space = spaces.Discrete(len(rest))
action_space = spaces.Tuple((next_node_space, charge_space, rest_space))
# n_actions = env.action_space.n
# print(n_actions)

print(action_space)
random_next_node = np.random.choice(next_node)
if random_next_node in [1,2,3]:
    random_charge = np.random.choice(charge)
    action = (random_next_node, random_charge, 0)
else:
    random_rest = np.random.choice(rest)
    action = (random_next_node, 0, random_rest)

print(action)

# print(observation_space.sample())
