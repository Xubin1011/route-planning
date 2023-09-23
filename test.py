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
# 初始化一个包含多行的二维列表
sorted_indices_list = [
    [0, 1, 2],
    [3, 4, 5],
    [None, 7, 8],
    [9, 10],
    [11, 12, 13, 14]
]

# 查询第三行的所有元素
third_row = sorted_indices_list[2]

# 打印第三行
for element in third_row:
    print(element)
