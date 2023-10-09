import torch

# 创建一个包含单个浮点数的张量
tensor = torch.tensor([3.])

# 使用item()方法获取单个元素并转换为整数
integer_value = int(tensor.item())

print(integer_value)  # 输出 3