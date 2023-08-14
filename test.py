import pandas as pd

# 读取CSV文件
data = pd.read_csv("cs_combo_bbox.csv")

# 给定的坐标
given_latitude = 0  # 用实际的值替代
given_longitude = 0  # 用实际的值替代

# 根据给定的坐标进行匹配
matching_row = data[
    (data["Latitude"] == given_latitude) & (data["Longitude"] == given_longitude)
]

if not matching_row.empty:
    elevation = matching_row["Elevation"].values[0]
    power = matching_row["Power"].values[0]
    print(f"Latitude: {given_latitude}, Longitude: {given_longitude}")
    print(f"Elevation: {elevation}, Power: {power}")
else:
    print("Coordinates not found in the dataset")



