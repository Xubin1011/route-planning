import pandas as pd
import numpy as np

# 目标坐标
x_target = 52.5253
y_target = 13.3694

# 数据框示例
data = {
    'Latitude': [x1, x2, x3, ...],  # 填充实际数据
    'Longitude': [y1, y2, y3, ...],  # 填充实际数据
    'Elevation': [e1, e2, e3, ...],  # 填充实际数据
    'Power': [p1, p2, p3, ...]  # 填充实际数据
}

initial_data_ch = pd.DataFrame(data)


# 计算每个数据点与目标坐标之间的Haversine距离
def haversine(x1, y1, x2, y2):
    # 将经纬度从度数转换为弧度
    x1, y1, x2, y2 = np.radians([x1, y1, x2, y2])

    # Haversine距离计算
    dlon = y2 - y1
    dlat = x2 - x1
    a = np.sin(dlat / 2) ** 2 + np.cos(x1) * np.cos(x2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # 地球的平均半径，单位：千米
    distance = c * r

    return distance


# 初始化最小距离和最近点的索引
min_distance = None
closest_index = None

# 查找距离最近的点
for index, row in initial_data_ch.iterrows():
    distance = haversine(x_target, y_target, row['Latitude'], row['Longitude'])
    if min_distance is None or distance < min_distance:
        min_distance = distance
        closest_index = index

# 获取距离最近的点的数据
closest_point = initial_data_ch.loc[closest_index]

print(closest_point)
