# Date: 7/24/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...


import pandas as pd
import heapq

# 导入distance_haversine.py中的haversine函数
from distance_haversine import haversine

# 读取CSV文件并提取经纬度信息
data = pd.read_csv("test.csv")
latitudes = data["Latitude"]
longitudes = data["Longitude"]

# 当前位置经纬度
x1, y1 = 49.176492, 9.231113

# 优先级队列，用于存储最近的3个地点的信息
closest_locations = []

# 遍历所有地点
for lat, lon in zip(latitudes, longitudes):
    # 计算当前位置与当前地点的距离
    distance = haversine(x1, y1, lat, lon)

    # 如果队列中地点数量小于3，直接插入当前地点
    if len(closest_locations) < 3:
        heapq.heappush(closest_locations, (distance, lat, lon))
    else:
        # 获取当前队列中距离最远的地点的距离
        max_distance, _, _ = closest_locations[2]

        # 如果当前地点距离更近，则替换最远的地点
        if distance < max_distance:
            heapq.heappop(closest_locations)
            heapq.heappush(closest_locations, (distance, lat, lon))

# 将结果转换为DataFrame
closest_locations = pd.DataFrame(closest_locations, columns=["Distance", "Latitude", "Longitude"])

print(closest_locations)
