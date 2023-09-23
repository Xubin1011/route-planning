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
import sys

# 获取命令行参数，sys.argv[0] 是脚本的名称，sys.argv[1] 是第一个参数，以此类推
if len(sys.argv) > 1:
    try_numbers = int(sys.argv[1])
else:
    print("No value for try_numbers provided.")
    sys.exit(1)

# 现在，您可以在脚本中使用try_numbers变量
print(f"Received try_numbers: {try_numbers}")

