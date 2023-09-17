import torch
import torch.nn as nn
import torch.nn.functional as F
from dqn_n_pois import DQN
import numpy as np
from environment_n_pois import rp_env

weights_path ="weights_033.pth"
# Initialization of state, Q-Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = rp_env()
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
n_observations = len(state)
n_actions = env.df_actions.shape[0]
q_network = DQN(n_observations, n_actions).to(device)
# Load weigths
checkpoint = torch.load(weights_path)
q_network.load_state_dict(checkpoint['model_state_dict'])



with torch.no_grad():
    q_values = Q_network(state_tensor)

q_values = q_values.cpu().numpy()  # 转换为numpy数组
sorted_indices = np.argsort(q_values[0])[::-1]  # 从大到小排序

for i, index in enumerate(sorted_indices):
    sorted_q_value = q_values[0, index]
    print(f"Action {index}: Q-value = {sorted_q_value}")

##otain next state  step()