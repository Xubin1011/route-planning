import pandas as pd
import numpy as np
import networkx as nx
import os
import random

from dijkstra_graph import haversine, x_source, y_source, x_target, y_target, max_edge_length
from consumption_duration import consumption_duration
import folium
from bounding_box import bbox

m = 13500 #(Leergewicht)
g = 9.81
rho = 1.225
A_front = 10.03
c_r = 0.01
c_d = 0.7
a = 0
eta_m, eta_battery = 0.8, 0.8

route_path = f'dij_path_{max_edge_length/1000}.csv'
weights_path = f'dijkstra_edges_{max_edge_length/1000}.csv'
map_name = f'dij_path_{max_edge_length/1000}.html'
stay_list = [0]

# select the closest node in graph G
def get_closest_node(G, latitude, longitude):
    closest_node = None
    closest_distance = float('inf')

    for node in G.nodes(data=True):
        node_latitude = node[1]['latitude']
        node_longitude = node[1]['longitude']

        # 计算给定节点与目标经纬度坐标之间的距离
        distance = haversine(latitude, longitude, node_latitude, node_longitude)

        if distance < closest_distance:
            closest_node = node[0]
            closest_distance = distance

    return closest_node

# cheack edges
def check_edge(x_current, y_current,ati_current, x_next, y_next, ati_next, power_next):
    global t_stay, t_secd_current, t_secch_current, stay_list
    terminated = False
    consumption, typical_duration, distance_meters = consumption_duration(x_current, y_current, ati_current, x_next, y_next, ati_next, m, g, c_r, rho, A_front, c_d, a, eta_m, eta_battery)
    t_stay = consumption / power_next *3600 # in s
    stay_list.append(t_stay)
    # the time that arriving next location
    t_arrival = t_secd_current + t_secch_current + typical_duration
    # the depature time when leave next location
    t_departure = t_arrival + t_stay

    ##################################################################
    # check rest, driving time constraint
    if t_arrival >= section:  # A new section begin before arrival next state, only consider the  last section
        t_secd_current = t_arrival % section
        if t_secch_current < min_rest:
            terminated = True
            print("Terminated: Violated max_driving times")
        t_secch_current = t_stay
    else:  # still in current section when arriving next poi
        if t_departure >= section:  # A new section begin before leaving next state,only consider the  last section
            t_secch_current += t_departure % section
            if t_secch_current < min_rest:
                terminated = True
                print("Terminated: Violated max_driving times")
            t_secch_current += (t_stay - t_departure % section)
            t_secd_current = 0
        else:  # still in current section
            t_secch_current = t_stay + t_secch_current
            t_secd_current +=  typical_duration
    return terminated

def check_path(path):
    path_lat = []
    path_lon = []
    path_alt = []
    path_power = []
    global stay_list
    for i in range(len(path)):
        Latitude, Longitude, Elevation, Power = pois_df.iloc[path[i]]
        path_lat.append(Latitude)
        path_lon.append(Longitude)
        path_alt.append(Elevation)
        path_power.append(Power)
        # print(path_lat,path_lon, path_alt, path_power)
    for t in range (len(path) - 1):
        terminated = check_edge(path_lat[t],path_lon[t], path_alt[t], path_lat[t+1], path_lon[t+1], path_alt[t+1], path_power[t+1])
        if terminated:
            unfeasible = True
            stay_list = []
            break
        else:
            unfeasible = False
    return unfeasible

def visualization(file1, file2, file3, source_lat, source_lon, target_lat, target_lon, map_name):
    # calculate the bounding box
    south_lat, west_lon, north_lat, east_lon = bbox(source_lat, source_lon, target_lat, target_lon)
    # Read data from file1
    data1 = pd.read_csv(file1)
    # Read data from file2
    data2 = pd.read_csv(file2)
    # Calculate the center of the bounding box
    center_lat = (south_lat + north_lat) / 2
    center_lon = (west_lon + east_lon) / 2
    # Create a map object
    map_object = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    # Add the bounding box area to the map
    bbox_coords = [[south_lat, west_lon], [north_lat, west_lon], [north_lat, east_lon], [south_lat, east_lon], [south_lat, west_lon]]
    folium.Polygon(locations=bbox_coords, color='blue', fill=True, fill_color='blue', fill_opacity=0.2).add_to(map_object)
    # Read data from file1 and add yellow markers for data points
    for _, row in data1.iterrows():
        # latitude, longitude = row['Latitude'], row['Longitude']
        # popup_content = 'Charging Station'
        # tooltip_content = f'Latitude: {latitude}, Longitude: {longitude}'
        # folium.Marker(location=[latitude, longitude], popup=popup_content, tooltip=tooltip_content, icon=folium.Icon(color='purple')).add_to(map_object)
        latitude, longitude = row['Latitude'], row['Longitude']
        folium.CircleMarker(location=[latitude, longitude], radius=1, color='yellow', fill=True, fill_color='yellow').add_to(map_object)
    # Read data from file2 and add blue markers for data points
    for _, row in data2.iterrows():
        latitude, longitude = row['Latitude'], row['Longitude']
        folium.CircleMarker(location=[latitude, longitude], radius=1, color='blue', fill=True, fill_color='blue').add_to(map_object)
    # Read data from file3 as path coordinates
    path_data = pd.read_csv(file3)
    path_coords = list(zip(path_data['Latitude'], path_data['Longitude']))
    path_infos = list(zip(path_data['Latitude'], path_data['Longitude'], path_data['Stay']))
    for coord in path_infos:
        latitude, longitude, stay = coord
        folium.Marker(location=[latitude, longitude],
                      popup=f'Latitude: {latitude}<br>Longitude: {longitude}<br>Stay: {stay/60}mins',
                      icon=folium.Icon(color='green')).add_to(map_object)
        # folium.CircleMarker(location=[latitude, longitude], radius=2, color='red', fill=True, fill_color='red').add_to(
        #     map_object)
    # Add a red line to represent the path
    folium.PolyLine(locations=path_coords, color='red').add_to(map_object)
    # Save the map as an HTML file and display it
    map_object.save(map_name)

#visualization(cs_path, p_path, route_path, myway.x_source, myway.y_source, myway.x_target, myway.y_target)
cs_path = 'cs_combo_bbox.csv'
p_path = 'parking_bbox.csv'
def visu(path):
    if os.path.exists(route_path):
        os.remove(route_path)
    path_lat = []
    path_lon = []
    global stay_list
    # print(stay_list)
    for i in range(len(path)):
        Latitude, Longitude, Elevation, Power = pois_df.iloc[path[i]]
        path_lat.append(Latitude)
        path_lon.append(Longitude)
    geo_coord = pd.DataFrame({'Latitude': path_lat, 'Longitude': path_lon, 'Stay': stay_list})
    geo_coord.to_csv(route_path, index=False)
    visualization(cs_path, p_path, route_path, x_source, y_source, x_target, y_target, map_name)

###########################################


t_stay, t_secd_current, t_secch_current = 0, 0, 0
# Each section has the same fixed travel time
min_rest = 2700  # in s
max_driving = 16200  # in s
section = min_rest + max_driving

# load weights
weights_df = pd.read_csv(weights_path)
weights_matrix = weights_df.values

# load pois
pois_df = pd.read_csv('dijkstra_pois.csv')
latitude = pois_df['Latitude'].values
longitude = pois_df['Longitude'].values

# creat graph
G = nx.DiGraph()

# add pois into graph
for i, row in pois_df.iterrows():
    G.add_node(i, latitude=row['Latitude'], longitude=row['Longitude'])

source = get_closest_node(G, x_source, y_source)
target = get_closest_node(G, x_target, y_target)
print("source:", source)
print("target:", target)

# add weights into graph
for i in range(len(pois_df)):
    for j in range(len(pois_df)):
        weight = weights_matrix[i, j]
        if weight != np.inf:
            G.add_edge(i, j, weight=weight)

# find the shortest_path and check it
k = 100
for i in range(k):
    shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight', method="dijkstra")
    unfeasible = check_path(shortest_path)
    print(f"shortest_path {shortest_path}, unfeasible is {unfeasible}")
    if unfeasible:
        print(f"shortest_path {shortest_path}, unfeasible is {unfeasible}")
        # randomly delete a vertex in the shortest path
        if len(shortest_path) > 2:
            remove_node = random.choice(shortest_path[1:-1])  # 随机选择一个中间节点
            print(remove_node)
            G.remove_node(remove_node)
        if i == k - 1:
            print(f"can not find a feasible path in {k}-shortest paths")
        continue
    total_cost = sum(G[shortest_path[i]][shortest_path[i + 1]]['weight'] for i in range(len(shortest_path) - 1))
    print(total_cost)
    break
visu(shortest_path)







