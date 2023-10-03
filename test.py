import torch

# 假设state是一个PyTorch张量
state = torch.tensor([[5.0000e+00, 6.9700e+02, 0.0000e+00, 1.5533e+04, 0.0000e+00, 0.0000e+00, 1.1403e+04]], device='cuda:0')

# 获取前两个值和第四个值
first_two_and_fourth_values = (state[0, 0], state[0, 1], state[0, 3])

print(first_two_and_fourth_values)
node, index, t_stay = list(first_two_and_fourth_values)
print(node, index, t_stay)
print(int(node), int(index), float(t_stay))