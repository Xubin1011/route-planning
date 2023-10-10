import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
import folium
from env_deploy import rp_env
from way_noloops import way
from global_var import initial_data_p, initial_data_ch, data_p, data_ch


env = rp_env()
myway = way()
#########################################################
# try_number = 47
##############Linux##################
weights_path ="/home/utlck/PycharmProjects/Tunning_results/weights_050_1000epis.pth"
route_path = f"/home/utlck/PycharmProjects/Tunning_results/dqn_route_050_1000epis.csv"
map_name = f"/home/utlck/PycharmProjects/Tunning_results/dqn_route_050_1000epis.html"

##############Win10#################################
# weights_path =f"G:\Tuning_results\weights_047_101.pth"
# route_path = f"G:\Tuning_results\dqn_route_047_101.csv"
# map_name = f"G:\Tuning_results\dqn_route_047_101.html"

cs_path = "cs_combo_bbox.csv"
p_path = "parking_bbox.csv"

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
def geo_coord(node, index):
    if node in range(myway.n_ch):
        Latitude, Longitude, Elevation, Power = initial_data_ch.iloc[index]
        return Latitude, Longitude, Elevation, Power
    else:
        Latitude, Longitude, Altitude = initial_data_p.iloc[index]
        power = None
        return Latitude, Longitude, Altitude, power

def save_pois(node, x, y, t_stay, power):
    try:
        df = pd.read_csv(route_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Latitude", "Longitude", "Stay", "Power"])
    # save new location
    new_row = {"Node": node, "Latitude": x, "Longitude": y, "Stay": t_stay, "Power": power}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(route_path, index=False)
#########################################################
def clear_route():
    try:
        df = pd.read_csv(route_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Node", "Latitude", "Longitude", "Stay", "Power"])
    df = pd.DataFrame(columns=["Node", "Latitude", "Longitude", "Stay", "Power"])
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
    # print("sorted_indices_list = ", sorted_indices_list)
    return(sorted_indices_list)

##############################################################
def bbox(source_lat, source_lon, target_lat, target_lon):
    # Calculate the North latitude, West longitude, South latitude, and East longitude
    south_lat = min(source_lat, target_lat)
    west_lon = min(source_lon, target_lon)
    north_lat = max(source_lat, target_lat)
    east_lon = max(source_lon, target_lon)

    return south_lat, west_lon, north_lat, east_lon

def visualization(cs_path, p_path, route_path, source_lat, source_lon, target_lat, target_lon, map_name):
    # calculate the bounding box
    south_lat, west_lon, north_lat, east_lon = bbox(source_lat, source_lon, target_lat, target_lon)
    # Read data from cs_path
    data1 = pd.read_csv(cs_path)
    # Read data from p_path
    data2 = pd.read_csv(p_path)
    # Calculate the center of the bounding box
    center_lat = (south_lat + north_lat) / 2
    center_lon = (west_lon + east_lon) / 2
    # Create a map object
    map_object = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    # Add the bounding box area to the map
    bbox_coords = [[south_lat, west_lon], [north_lat, west_lon], [north_lat, east_lon], [south_lat, east_lon], [south_lat, west_lon]]
    folium.Polygon(locations=bbox_coords, color='blue', fill=True, fill_color='blue', fill_opacity=0.2).add_to(map_object)
    # Read data from cs_path and add yellow markers for data points
    for _, row in data1.iterrows():
        latitude, longitude = row['Latitude'], row['Longitude']
        folium.CircleMarker(location=[latitude, longitude], radius=1, color='yellow', fill=True, fill_color='yellow').add_to(map_object)
    # Read data from p_path and add blue markers for data points
    for _, row in data2.iterrows():
        latitude, longitude = row['Latitude'], row['Longitude']
        folium.CircleMarker(location=[latitude, longitude], radius=1, color='blue', fill=True, fill_color='blue').add_to(map_object)
    # Read data from route_path as path coordinates
    path_data = pd.read_csv(route_path)
    path_coords = list(zip(path_data['Latitude'], path_data['Longitude']))
    path_infos = list(zip(path_data['Node'],path_data['Latitude'], path_data['Longitude'], path_data['Stay'], path_data['Power']))

    node_sor, sor_lat, sor_lon, sor_stay, power_sor = path_infos[0]  #souce
    node_target, tar_lat, tar_lon, tar_stay, power_target = path_infos[-1] #target

    for coord in path_infos:
        node_attr, latitude, longitude, stay, power = coord
        if latitude == sor_lat and longitude == sor_lon:
            folium.Marker(location=[latitude, longitude],
                          popup=f'Node: {node_attr}<br>Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay / 60}mins<br>Power: {power}kWh',
                          icon=folium.Icon(color='red')).add_to(map_object)
        if stay != 0:
            folium.Marker(location=[latitude, longitude],
                          popup=f'Node: {node_attr}<br>Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay / 60}mins<br>Power: {power}kWh',
                          icon=folium.Icon(color='green')).add_to(map_object)
        if latitude == tar_lat and longitude == tar_lon:
            folium.Marker(location=[latitude, longitude],
                          popup=f'Node: {node_attr}<br>Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay / 60}mins<br>Power: {power}kWh',
                          icon=folium.Icon(color='red')).add_to(map_object)

    # cs_html_icon = folium.DivIcon(html=f'<div style="font-size: 16px; color: yellow;">CS</div>')
    # cs_html_icon_source = folium.DivIcon(html=f'<div style="font-size: 16px; color: red;">CS_S</div>')
    # cs_html_icon_target = folium.DivIcon(html=f'<div style="font-size: 16px; color: red;">CS_T</div>')
    # p_html_icon = folium.DivIcon(html=f'<div style="font-size: 16px; color: blue;">P</div>')
    # p_html_icon_source = folium.DivIcon(html=f'<div style="font-size: 16px; color: red;">P_S</div>')
    # p_html_icon_target = folium.DivIcon(html=f'<div style="font-size: 16px; color: red;">P_T</div>')
    # for coord in path_infos:
    #     node_attr, latitude, longitude, stay = coord
    #     if int(node_attr) in range(0, 6): # CS
    #         if latitude == sor_lat and longitude == sor_lon:
    #             folium.Marker(location=[latitude, longitude],
    #                           popup=f'Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay / 60}mins',
    #                           icon=cs_html_icon_source).add_to(map_object)
    #         if stay != 0:
    #             folium.Marker(location=[latitude, longitude],
    #                           popup=f'Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay/60}mins',
    #                           icon=cs_html_icon).add_to(map_object)
    #         if latitude == tar_lat and longitude == tar_lon:
    #             folium.Marker(location=[latitude, longitude],
    #                           popup=f'Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay / 60}mins',
    #                           icon=cs_html_icon_target).add_to(map_object)
    #     else: #P
    #         if latitude == sor_lat and longitude == sor_lon:
    #             folium.Marker(location=[latitude, longitude],
    #                           popup=f'Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay / 60}mins',
    #                           icon=p_html_icon_source).add_to(map_object)
    #         if stay != 0:
    #             folium.Marker(location=[latitude, longitude],
    #                           popup=f'Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay/60}mins',
    #                           icon=p_html_icon).add_to(map_object)
    #         if latitude == tar_lat and longitude == tar_lon:
    #             folium.Marker(location=[latitude, longitude],
    #                           popup=f'Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay / 60}mins',
    #                           icon=p_html_icon_target).add_to(map_object)

        # folium.CircleMarker(location=[latitude, longitude], radius=2, color='red', fill=True, fill_color='red').add_to(
        #     map_object)
    # Add a red line to represent the path
    folium.PolyLine(locations=path_coords, color='red').add_to(map_object)
    # Save the map as an HTML file and display it
    map_object.save(map_name)
#############################################################

# Initialization of state, Q-Network, state history list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clear_route()

state, info = env.reset()
# state = [9, 5, 0.8, 0, 0, 0, 0]  # test
n_observations = len(state)
node_current, index_current, soc, t_stay, t_secd_current, t_secp_current, t_secch_current = state
x_current, y_current, alti_current, power = myway.geo_coord(node_current, int(index_current))
# save_pois(state)

initial_state = state
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
print("reseted state = ", state)
state_history = []# save state tensor in a list
state_history.append(state)
# print("state history = ", state_history)

n_actions = env.df_actions.shape[0]
policy_net = DQN(n_observations, n_actions).to(device)

# Load weigths
checkpoint = torch.load(weights_path)
# print(checkpoint)
policy_net.load_state_dict(checkpoint)
# print("policy_net:", policy_net)
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

    # check actions one by one from the largest q value to the smallest q value
    # until obtain an action that does not violate constraint
    # If no feasible action, take a step back
    for t in range(n_actions):
        # Set the checked actions to None
        action = sorted_indices_list[-1][t]
        sorted_indices_list[-1][t] = None  # delete accepted action
        print("sorted_indices_list[-1] =", sorted_indices_list[-1])
        if action == None:
            continue
        else:
            observation, terminated, d_next = env.step(action)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            if terminated == False: #accept action
                print(f"******The action {action} in step {num_step} is selected\n")
                num_step += 1
                state = next_state
                state_history.append(next_state)
                break
            else:
                if d_next <= 25000: # Arrival target
                    state_history.append(next_state)
                    target_flag = True
                    print("******Arrival target\n")
                    break
                else:
                    # violate contraints
                    if t == n_actions - 1: # all q-values have been checked, disable last state,
                        del sorted_indices_list[num_step] #  state values are deleted from list
                        num_step -= 1
                        step_back = True
                        del state_history[num_step] # delete last state
                        print(f"******no feasible action found in step {num_step}, take a step back\n")
                        break

    if i == max_steps - 1 and not target_flag:
        print(f"After {max_steps} steps no feasible route")
        break

    if target_flag == True:
        print(f"Finding a  feasible route after {i+1} steps")
        print("State history:\n", state_history)
        print("sorted_indices_list\n: ", sorted_indices_list)
        for state in state_history:
            #state = (node, index, soc, t_stay, t_secd, t_secr, t_secch)
            first_two_and_fourth_values = (state[0, 0], state[0, 1], state[0, 3])
            node, index, t_stay = list(first_two_and_fourth_values)
            x, y, _, power, = geo_coord(int(node), int(index))
            save_pois(int(node), x, y, float(t_stay), power)
        visualization(cs_path, p_path, route_path, myway.x_source, myway.y_source, myway.x_target, myway.y_target, map_name)
        break

    if num_step < 0:
        print(f"No feasible route from initial state {initial_state}")
        break

print("done")


        
        
            

