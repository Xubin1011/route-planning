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
# import numpy as np
# next_node = np.array([1, 2, 3, 4, 5])
# charge = np.array([0, 0.3, 0.5, 0.8])
# rest = np.array([0, 0.3, 0.6, 0.9, 1])
# next_node_space = spaces.Discrete(len(next_node))
# charge_space = spaces.Discrete(len(charge))
# rest_space = spaces.Discrete(len(rest))
# action_space = spaces.Tuple((next_node_space, charge_space, rest_space))
# n_actions = env.action_space.n
# print(n_actions)
action_space = env.action_space
space_sizes = [component.n for component in action_space]
print(action_space)
print(space_sizes)


for random_index in range(np.prod(space_sizes)):
    # 初始化一个列表来存储每个组件的动作
    selected_actions = []

    # 将索引映射到各个组件的动作
    for size in reversed(space_sizes):
        selected_actions.append(random_index % size)
        random_index //= size

    selected_actions.reverse()  # 因为从后往前添加的，所以需要反转顺序

    # 输出索引及其对应的动作
    print("Index:", random_index, "Selected Actions:", selected_actions)


# print(observation_space.sample())
