# import pandas as pd
#
# def check_max_power_smaller_than_rated_output(file_path):
#     # Read the CSV file into DataFrame
#     df = pd.read_csv(file_path)
#
#     # Check if "Max_socket_power" values are smaller than "Rated_output" values
#     for index, row in df.iterrows():
#         power = row['Power']
#         # rated_output = row['Rated_output']
#
#         # if max_socket_power > rated_output:
#         #      print(f"Row {index + 1}: Max_socket_power value ({max_socket_power}) is not smaller than Rated_output value ({rated_output}).")
#         if power > 100:
#             print(index+1)
#         # else:
#         #     print("none")
#
#
# # Example usage
# file_path = 'cs_filtered_03.csv'
# check_max_power_smaller_than_rated_output(file_path)

import pandas as pd

# 读取Excel文件
file_path = "Ladesaeulenregister-processed.xlsx"
df = pd.read_excel(file_path)

# 初始化一个空列表来存储包含整数值的行号
rows_with_integer_values = []

# 遍历DataFrame的第一列（Latitude）和第二列（Longitude）
for index, row in df.iterrows():
    latitude = row["Latitude"]
    longitude = row["Longitude"]

    # 检查是否为整数
    if isinstance(latitude, int) or isinstance(longitude, int):
        rows_with_integer_values.append(index + 1)  # 行号从1开始，所以需要加1

# 输出包含整数值的行号
if rows_with_integer_values:
    print("整数值出现在以下行：")
    print(rows_with_integer_values)
else:
    print("未找到包含整数值的行。")

