import torch
import torch.nn as nn
import torch.nn.functional as F
from dqn_n_pois import DQN
import numpy as np
from environment_n_pois import rp_env

env = rp_env()
state, info = env.reset()
n_observations = len(state)
n_actions = 22
Q_network = DQN(n_observations, n_actions).to(device)
weights = torch.load("weights_path")
Q_network.load_state_dict(weights['model_state_dict'])
model.eval()

state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    q_values = Q_network(state_tensor)

q_values = q_values.cpu().numpy()  # 转换为numpy数组
sorted_indices = np.argsort(q_values[0])[::-1]  # 从大到小排序

for i, index in enumerate(sorted_indices):
    sorted_q_value = q_values[0, index]
    print(f"Action {index}: Q-value = {sorted_q_value}")

##otain next state  step()