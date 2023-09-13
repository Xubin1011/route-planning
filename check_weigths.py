import torch
import torch.nn as nn
import torch.nn.functional as F
from dqn import DQN
import numpy as np
from environment import rp_env
pth_file_path = "weights_010.pth"
n_observations = 8
n_actions = 22
Q_network = DQN(n_observations, n_actions)
checkpoint = torch.load(pth_file_path)
Q_network.load_state_dict(checkpoint['model_state_dict'])
print(Q_network)