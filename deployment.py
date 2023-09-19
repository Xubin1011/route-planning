import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dqn_n_actions import DQN
from env_n_actions import rp_env
from way_calculation import way
env = rp_env()
myway = way()
#########################################################
route_path = "route.csv"
def save_pois(x,y):
    try:
        df = pd.read_csv(route_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Latitude", "Longitude"])
    # save new location
    new_row = {"Latitude": x, "Longitude": y}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(route_path, index=False)

#############################################################
actions_path = "actions.csv"
weights_path ="weights_037.pth"


# Initialization of state, Q-Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state, info = env.reset()
node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = state
save_pois(x_current, y_current)
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

##############################################################
# check each action from the largest q value to the smallest q value
# until obtain an action that terminated is false
for i in range(n_actions):
    action = sorted_indices[i]
    print(f"The action {i} with q value {sorted_q_values[i]} is selected")
    observation, reward, terminated = env.step(action)
    node_next, x_next, y_next, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = observation
    if terminated == False:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        save_pois(x_next, y_next)
        state = next_state
        break
    else:
        if x_next == myway.x_target and y_next == myway.y_target:
            save_pois(x_next, y_next)
            break
        else:
            if i == n_actions - 1:
                print(f"No feasible route found at ({x_next},{y_next})")
                break



        
        
            

