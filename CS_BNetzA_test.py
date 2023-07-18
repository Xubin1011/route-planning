import pandas as pd

# # 读取Excel文件
# file_path = 'Ladesaeulenregister.xlsx'
# df = pd.read_excel(file_path)

# # 直接在原始文件上删除前11行
# df.drop(df.index[:11], inplace=True)
#
# # 需要删除的列索引或列标签列表
# columns_to_drop = df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 13, 16, 19, 22, 25]]
#
# # 删除指定列
# df.drop(columns=columns_to_drop, inplace=True)
#
# # 保存修改后的数据到原始文件
# df.to_excel(file_path, index=False)

# # 指定要查看类型的列索引
# columns_to_check = [3, 5, 7, 9]
#
# for col_index in columns_to_check:
#     col_name = df.columns[col_index]
#     unique_values_counts = df.iloc[:, col_index].value_counts()
#     print(f"列 '{col_name}' 中不同类型及其出现次数：")
#     print(unique_values_counts)
#     print("--------------------------------------")

# # 指定要比较的列索引
# columns_to_compare = [2, 4, 6, 8, 10]
#
# # 获取每行的最大值并保存在第3列
# df['Max_Value'] = df.iloc[:, columns_to_compare].max(axis=1)
#
# # 保存修改后的数据到新文件
# output_file_path = 'Ladesaeulenregister_max_values.xlsx'
# df.to_excel(output_file_path, index=False)
#
# print("每行最大值已保存到新文件:", output_file_path)


# 读取Excel文件
file_path_max_values = 'Ladesaeulenregister_max_values.xlsx'
df_max_values = pd.read_excel(file_path_max_values)

# 获取第3列和第12列的数据
column_3 = df_max_values.iloc[:, 2]  # 第3列，索引从0开始
column_12 = df_max_values.iloc[:, 11]  # 第12列，索引从0开始

# 比较两列内容是否完全一样
are_columns_equal = column_3.equals(column_12)

if are_columns_equal:
    print("第3列和第12列内容完全一样。")
else:
    print("第3列和第12列内容不完全一样。")
    # 找出不一样的行
    different_rows = df_max_values[column_3 != column_12]
    print("不一样的行：")
    print(different_rows)