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


# 创建一个示例 DataFrame
data_p = pd.DataFrame({'Latitude': [51.4958, 51.4960, 51.4962],
                       'Longitude': [12.4970, 12.4972, 12.4974]})

# 打印原始 DataFrame
print("原始 DataFrame:")
print(data_p)

# 要删除的索引列表
delete_indexes = [0, 2]

# 使用 drop 方法删除指定的行
data_p = data_p.drop(delete_indexes)

# 打印删除后的 DataFrame
print("\n删除后的 DataFrame:")
print(data_p)

import pandas as pd

# 创建一个示例 DataFrame
data_p = pd.DataFrame({'Latitude': [51.4958, 51.4960, 51.4962, 51.4960],
                       'Longitude': [12.4970, 12.4972, 12.4974, 12.4970]})

# 打印原始 DataFrame
print("原始 DataFrame:")
print(data_p)

# 指定要查找的经纬度值
x = 51.4960
y = 12.4970

# 使用条件筛选获取匹配的索引
matching_indexes = data_p[(data_p["Latitude"] == x) & (data_p["Longitude"] == y)].index

# 打印匹配的索引
print("\n匹配的索引:")
print(matching_indexes)

# 使用 drop 方法删除指定的行
data_p = data_p.drop(1)

# 打印删除后的 DataFrame
print("\n删除后的 DataFrame:")
print(data_p)
# 使用条件筛选获取匹配的索引
matching_indexes = data_p[(data_p["Latitude"] == x) & (data_p["Longitude"] == y)].index

# 打印匹配的索引
print("\n匹配的索引:")
print(matching_indexes)