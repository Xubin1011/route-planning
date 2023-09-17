import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dqn_n_pois import DQN
from environment_n_pois import rp_env

actions_path = "actions.csv"
weights_path ="weights_037.pth"
# Initialization of state, Q-Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = rp_env()
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
n_observations = len(state)
print("reseted state = ", state)

n_actions = env.df_actions.shape[0]
q_network = DQN(n_observations, n_actions).to(device)

# Load weigths
checkpoint = torch.load(weights_path)
q_network.load_state_dict(checkpoint['model_state_dict'])
print("q_network:", q_network)

q_network.eval()
q_values = q_network(state)
print("q_values =", q_values)

# Sort Q values from large to small
sorted_q_values, sorted_indices = torch.sort(q_values, descending=True)

# check each action from the largest q value to the smallest q value
# until obtain an action that terminated is false

df = pd.read_csv(actions_path)
action =

