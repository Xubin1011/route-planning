import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
# from dqn_n_actions import DQN
from env_deploy import rp_env
from way_calculation import way
from visualization import visualization

env = rp_env()
myway = way()
#########################################################
# actions_path = "actions.csv"
weights_path ="/home/utlck/PycharmProjects/Tunning_results/weights_040.pth"
cs_path = "cs_combo_bbox.csv"
p_path = "parking_bbox.csv"
route_path = "route.csv"

class DQN(nn.Module):

    #Q-Network with 2 hidden layers, 128 neurons
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Forward propagation with ReLU
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def save_pois(x, y, t_stay):
    try:
        df = pd.read_csv(route_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Latitude", "Longitude", "Stay"])
    # save new location
    new_row = {"Latitude": x, "Longitude": y, "Stay": t_stay}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(route_path, index=False)
#########################################################
def clear_route():
    try:
        df = pd.read_csv(route_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Latitude", "Longitude", "Stay"])
    df = pd.DataFrame(columns=["Latitude", "Longitude", "Stay"])
    df.to_csv(route_path, index=False)

#########################################################
# save all outputs from Q-Network
sorted_indices_list = [] # This list saving all outputs from Q-Network
def save_q(state):
    # obtaion q values
    q_values = policy_net(state)
    print("q_values =", q_values)
    # Sort Q values from large to small
    sorted_q_values, sorted_indices = torch.sort(q_values, descending=True)
    # Save the sorted_indices in list
    sorted_indices_list.append(sorted_indices[0].tolist())
    print("sorted_indices_list = ", sorted_indices_list)
    return(sorted_indices_list)

##############################################################
# # check an action, update flags, save accept pois
# def check_acts(action):
#     observation, terminated, d_next = env.step(action)
#     # node_next, x_next, y_next, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = observation
#     next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
#
#     if terminated == False:  # accept action
#         # save_pois(x_next, y_next, t_stay)
#         step_flag = False
#     else:
#         step_flag = True # Violate constrains
#         # if x_next == myway.x_target and y_next == myway.y_target:
#         if d_next <= 25000:  # arrival target, accept action
#             # save_pois(x_next, y_next, t_stay)
#             target_flag = True
#
#     return (next_state, step_flag, target_flag)
#############################################################

# Initialization of state, Q-Network, state history list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clear_route()

state, info = env.reset()
n_observations = len(state)
node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = state
save_pois(x_current, y_current, t_stay)
initial_state = state
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
print("reseted state = ", state)
state_history = []# save state tensor in a list
state_history.append(state)
print("state history = ", state_history)

n_actions = env.df_actions.shape[0]
policy_net= DQN(n_observations, n_actions).to(device)

# Load weigths
checkpoint = torch.load(weights_path)
print(checkpoint)
# policy_net.load_state_dict(checkpoint['model_state_dict'])
policy_net.load_state_dict(checkpoint)
print("policy_net:", policy_net)
policy_net.eval()

num_step = 0
max_steps = 1000
# step_flag = False  # no terminated, "True": Violate constrains,terminated
target_flag = False # not arrival target
step_back = False
##################################################
# main loop
for i in range(0, max_steps): # loop for steps

    if step_back == False: # new output from Q network
        sorted_indices_list = save_q(state)

    # check actions from the largest q value to the smallest q value
    # until obtain an action that does not violate constraint
    # If no feasible action, take a step back
    for t in sorted_indices_list[-1]:

        # Set the checked actions to None
        action = t
        sorted_indices_list[-1][t] = None  # delete accepted action
        if action == None:
            continue
        else:
            observation, terminated, d_next = env.step(action)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            if terminated == False: #accept action
                print(f"The action {action} in step {num_step} is selected")
                num_step += 1
                state = next_state
                state_history.append(next_state)
                break
            else:
                if d_next <= 25000: # Arrival target
                    state_history.append(next_state)
                    target_flag = True
                    print("Arrival target")
                    break
                else:
                    # violate contraints
                    if t == len(sorted_indices_list[num_step]) - 1:
                        del sorted_indices_list[num_step]
                        num_step -= 1
                        step_back = True
                        del state_history[num_step]
                        print(f"no feasible action found in step {num_step}, take a step back ")
                        break

    if i == max_steps - 1 and not target_flag:
        print(f"After {max_steps} steps no feasible route")
        break

    if target_flag == True:
        print(f"Finding a  feasible route after {i+1} steps")
        for state in state_history:
            x1, y1, t_stay = state[1], state[2], state[4]
            save_pois(x1, y1, t_stay)
        visualization(cs_path, p_path, route_path, myway.x_source, myway.y_source, myway.x_target, myway.y_target)
        break

    if num_step < 0:
        print(f"No feasible route from initial state {initial_state}")
        break

print("done")


        
        
            

