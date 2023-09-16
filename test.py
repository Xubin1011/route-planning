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
from environment_n_pois import rp_env


df_actions = pd.read_csv("actions.csv")
print(df_actions)
action_space = spaces.Discrete(df_actions.shape[0])
print(action_space)
n_actions = df_actions.shape[0]
print(n_actions)