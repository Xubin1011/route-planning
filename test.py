import random
import sys

from nearest_location import nearest_location
from consumption_duration import consumption_duration
from consumption_duration import haversine
from way_calculation import way
from env_n_actions import rp_env
import pandas as pd

# env = rp_env()
# # x = 2.12345
# # y = 3.67890
# # loop = env.check_loop(x,y)
# # print(loop)
# env.clear_loop_file()

list = [[1,2,3],[4,5,6],[7,8,9]]
last = list[0][-1]
print(last)
