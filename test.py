import pandas as pd

# 读取CSV文件
df = pd.read_csv('cs_data.csv')

# 统计Rated_output列中不同值的个数
rated_output_counts = df['Rated_output'].value_counts()

# 将结果写入txt文件
output_filename = 'rated_output_counts.txt'
with open(output_filename, 'w') as f:
    f.write("Rated_output列中值的种类及个数:\n")
    f.write(rated_output_counts.to_string())

print("已将结果写入", output_filename)
