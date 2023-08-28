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


ef check_types(file_path_check_types):

    # Read the data from the Excel/csv file
    data = pd.read_csv(file_path_check_types)
    #data = pd.read_excel(file_path_check_types)

    # Calculate unique values and their occurrences in 'Socket1', 'Socket2', 'Socket3', 'Socket4'
    #socket_columns = ['Socket_1', 'Socket_2', 'Socket_3', 'Socket_4']
    socket_columns = ['Rated_output', 'Max_socket_power', 'Max_power'