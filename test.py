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

class MyData:
    def __init__(self):
        self.data_ch = pd.DataFrame({'Latitude': [1.0, 2.0, 3.0],
                                     'Longitude': [4.0, 5.0, 6.0],
                                     'Alt':[2,4,5]})


def test_data_ch_manipulation():
    # 创建一个示例并加载数据
    my_data = MyData()
    print(my_data.data_ch)

    # 设置要匹配的经纬度值
    next_x = 2.0  # 实际的经度值
    next_y = 5.0  # 实际的纬度值

    # 使用布尔索引筛选出要删除的行
    mask = (my_data.data_ch['Latitude'] != next_x) | (my_data.data_ch['Longitude'] != next_y)

    # 根据条件删除行
    my_data.data_ch = my_data.data_ch[mask]
    print(my_data.data_ch)



# 运行测试函数
test_data_ch_manipulation()
