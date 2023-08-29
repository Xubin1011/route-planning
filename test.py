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
# state, info = env.reset()
# n_observations = len(state)
#
#
# print(n_observations)

import itertools
import random

# 定义每个部分的可能取值
next_node_values = [1, 2, 3, 4, 5]
charge_space_values = [0, 0.3, 0.5, 0.8]
rest_space_values = [0, 0.3, 0.6, 0.9, 1]

# 生成所有可能的动作组合，根据条件进行过滤
all_action_combinations = []
for next_node in next_node_values:
    if next_node in [1, 2, 3]:
        valid_rest_values = [0]
    else:
        valid_rest_values = [0.3, 0.6, 0.9, 1]

    for charge in charge_space_



        values:
        for rest in valid_rest_values:
            all_action_combinations.append((next_node, charge, rest))

# 随机选择一个动作作为采样
sampled_action = random.choice(all_action_combinations)

print("Sampled Action:", sampled_action)
