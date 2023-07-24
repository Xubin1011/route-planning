# Date: 7/24/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...


import os

file_path = 'F:/OneDrive/Thesis/Code/route-planning/parking_location_noduplicate.csv'
if os.path.exists(file_path):
    print("文件存在。")
else:
    print("文件不存在。")
