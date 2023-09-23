import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dqn_n_actions import DQN
from env_deploy import rp_env
from way_calculation import way
from visualization import  visualization

env = rp_env()
myway = way()
#########################################################
# actions_path = "actions.csv"
weights_path ="weights_037.pth"
cs_path = "cs_combo_bbox.csv"
p_path = "parking_bbox.csv"
route_path = "route.csv"

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
# save all outputs from Q-Network
sorted_indices_list = [] # This list saving all outputs from Q-Network
def save_q(state):
    # obtaion q values
    q_values = q_network(state)
    print("q_values =", q_values)
    # Sort Q values from large to small
    sorted_q_values, sorted_indices = torch.sort(q_values, descending=True)
    # Save the sorted_indices in list
    sorted_indices_list.append(sorted_indices.clone())
    print(sorted_indices_list)
    return(sorted_indices_list)

##############################################################
# check an action from the largest q value to the smallest q value
# until obtain an action that does not violate constraint
# If no feasible action, take a step back
def check_acts(action):
    observation, terminated, d_next = env.step(action)
    node_next, x_next, y_next, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = observation
    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    if terminated == False:  # accept action
        save_pois(x_next, y_next, t_stay)
        target_flag = False
        sorted_indices_list[num_step][i] = None # delete accepted action

    else:
        # if x_next == myway.x_target and y_next == myway.y_target:
        if d_next <= 25000:  # arrival target, accept action
            save_pois(x_next, y_next, t_stay)
            target_flag = True
            sorted_indices_list[num_step][i] = None  # delete accepted action
            print("arrival target")

        else:  # not a feasible action
            sorted_indices_list[num_step][i] = None  # delete accepted action
            if i == n_actions - 1:
                target_flag = True
                no_feasible_actions = True
                del sorted_indices_list[num_step]
                print(f"No feasible route found at ({x_next},{y_next}), take a step back")
                # if step == 0:
                #     print(f"No feasible route found with initial state {initial_state}")

    return (next_state, target_flag, no_feasible_actions)
#############################################################
# Initialization of state, Q-Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state, info = env.reset()
node_current, x_current, y_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = state
save_pois(x_current, y_current, t_stay)
initial_state = state
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

num_step = 0
max_steps = 1000
target_flag = False
no_feasible_actions = False
##################################################
# main loop
for num_step in range(0, max_steps):
    sorted_indices_list = save_q(state)
    for i in sorted_indices_list[num_step]:
        action = sorted_indices_list[num_step][i]
        if action == None:
            continue
        else:
            print(f"The action {action} in step {num_step} is selected")
            next_state, target_flag, no_feasible_actions = check_acts(action)

    while not target_flag:

        if not no_feasible_actions:
            state = next_state
            num_step += 1
            if num_step == max_steps - 1:
                print(f"can not find target after {max_steps} steps")
                break
        else:
            num_step -= 1
            next_state, target_flag, no_feasible_actions = check_acts(num_step)


visualization(cs_path, p_path, route_path, myway.x_source, myway.y_source, myway.x_target, myway.y_target)

print("done")


        
        
            

