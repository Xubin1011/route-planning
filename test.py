import pandas as pd
import random

# 读取 CSV 文件
df = pd.read_csv("actions.csv")

# 获取随机索引
random_index = random.randint(0, len(df) - 1)
print(random_index)

# 从 DataFrame 中获取随机行
random_action = df.iloc[random_index]
a,b,c = random_action
print("Randomly selected action:")
print(random_action)
print(a)
print(b)
print(c)
