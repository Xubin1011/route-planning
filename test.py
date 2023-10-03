import pandas as pd
import numpy as np


    # 创建一个示例 DataFrame
data = pd.read_csv("cs_combo_bbox.csv")



# 测试从 DataFrame 中提取 'Latitude' 列的值
index = 1  # 假设 index 是有效的

latitude_value = np.float32(data.loc[index, 'Latitude'])
print(latitude_value)

