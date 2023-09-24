# Date: 2023/9/24
# Author: Xubin Zhang
# Description: This file contains the implementation of...
import pandas as pd
import numpy as np
import folium


#initialization
x_source = 49.0130 #source
y_source = 8.4093
x_target = 52.5253 #target
y_target = 13.3694
mass = 13500 #(Leergewicht) in kg
g = 9.81
rho = 1.225
A_front = 10.03
c_r = 0.01
c_d = 0.7
a = 0
eta_m = 0.82
eta_battery = 0.82

####################################################################
def bounding_box(source_lat, source_lon, target_lat, target_lon):
    # Calculate the North latitude, West longitude, South latitude, and East longitude
    south_lat = min(source_lat, target_lat)
    west_lon = min(source_lon, target_lon)
    north_lat = max(source_lat, target_lat)
    east_lon = max(source_lon, target_lon)
    return south_lat, west_lon, north_lat, east_lon
###################################################################
def haversine(x1, y1, x2, y2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = np.radians([x1, y1, x2, y2])

    # Haversine formula
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    # Earth's mean radius in kilometers
    radius = 6371.0
    distance = radius * c

    # Convert the distance to meters
    distance_meters = distance * 1000

    return distance_meters
################################################################
#Description: alpha is the slope
def calculate_alpha(x1, y1, c1, x2, y2, c2):
    # Calculate the haversine distance
    distance_meters = haversine(x1, y1, x2, y2)
    # Calculate sinalpha based on c2-c1
    elevation_difference = c2 - c1
    slope = np.arctan(elevation_difference / distance_meters)  # (slope in radians) slope belongs to -pi/2 to pi/2
    sin_alpha = np.sin(slope)
    cos_alpha = np.cos(slope)

    return sin_alpha, cos_alpha, distance_meters
###################################################################
#Description: Calculate consumption (the power needed for vehicle motion) between two POIs
# P_m  = v ( mgsinα + mgC_r cosα +  1/2  ρv^2 A_front C_d  + ma ) (in W)
# m : Mass of the vehicle (in kg)
# g :  Acceleration of gravity (in m/s^2)
# c_r : Coefficient of rolling resistance
# rho : Air density (in kg/m^3)
# A_front : Frontal area of the vehicle (in m^2)
# c_d :  Coefficient of drag
# a :  Acceleration (in m/s^2)
# eta_m: the energy efficiency of transmission, motor and power conversion
# eta_battery: the efficiency of transmission, generator and in-vehicle charger
def consumption_duration(x1, y1, c1, x2, y2, c2, m, g, c_r, rho, A_front, c_d, a, eta_m, eta_battery):
    sin_alpha, cos_alpha, distance_meters = calculate_alpha(x1, y1, c1, x2, y2, c2)
    average_speed = 100 * 1000 /3600 #in m/s
    typical_duration = distance_meters / average_speed # in s

    mgsin_alpha = m * g * sin_alpha
    mgCr_cos_alpha = m * g * c_r * cos_alpha
    air_resistance = 0.5 * rho * (average_speed ** 2) * A_front * c_d
    ma = m * a

    power = average_speed * (mgsin_alpha + mgCr_cos_alpha + air_resistance + ma) / eta_m

    # Recuperated energy
    if power < 0:
        if average_speed < 4.17: # 4.17m/s = 15km/h
            power = 0
        else:
            power = power * eta_battery
            if power < -150000:  # 150kW
                power = -150000
    consumption = power * typical_duration / 3600 / 1000  #(in kWh)
    return consumption, typical_duration, distance_meters #(in kWh, s, m)
###############################################################################

# Obtain the location of charging stations in 100 small bbox
def dijkstra_pois():
    df = pd.read_csv('cs_combo_bbox.csv')

    south_lat, west_lon, north_lat, east_lon = bounding_box(x_source, y_source, x_target, y_target)
    # 创建地图
    m = folium.Map(location=[(south_lat + north_lat) / 2, (west_lon + east_lon) / 2], zoom_start=10)
    dijkstra_pois = {}

    # 遍历每个小的bbox
    for i in range(10):
        for j in range(10):
            # 计算当前小bbox的边界框范围
            bbox_south_lat = south_lat + (north_lat - south_lat) * i / 10
            bbox_north_lat = south_lat + (north_lat - south_lat) * (i + 1) / 10
            bbox_west_lon = west_lon + (east_lon - west_lon) * j / 10
            bbox_east_lon = west_lon + (east_lon - west_lon) * (j + 1) / 10

            # 添加小的bbox范围
            folium.Rectangle(bounds=[(bbox_south_lat, bbox_west_lon), (bbox_north_lat, bbox_east_lon)],
                             color='blue').add_to(m)

            # 筛选出位于当前小bbox内的充电站数据
            bbox_filtered_df = df[(df['Latitude'] >= bbox_south_lat) &
                                  (df['Latitude'] <= bbox_north_lat) &
                                  (df['Longitude'] >= bbox_west_lon) &
                                  (df['Longitude'] <= bbox_east_lon)]


            if not bbox_filtered_df.empty:
                # 找到最大power的充电站
                max_power_stations = bbox_filtered_df[bbox_filtered_df['Power'] == bbox_filtered_df['Power'].max()]

                # 如果只有一个最大power的充电站，则将其坐标和power保存到dijkstra_pois中
                if len(bbox_filtered_df) == 1:
                    max_power_station = max_power_stations.iloc[0]
                    dijkstra_pois[(bbox_filtered_df['Latitude'].values[0],
                                   bbox_filtered_df['Longitude'].values[0])] = (max_power_station['Latitude'],
                                                                                max_power_station['Longitude'],
                                                                                max_power_station['Power'])
                else:
                    # 计算当前bbox中心点
                    bbox_center_lat = (bbox_south_lat + bbox_north_lat) / 2
                    bbox_center_lon = (bbox_west_lon + bbox_east_lon) / 2

                    # 找到距离bbox中心点最近的充电站
                    nearest_station = None
                    min_distance = float('inf')

                    for _, row in max_power_stations.iterrows():
                        station_lat = row['Latitude']
                        station_lon = row['Longitude']
                        station_distance = haversine(bbox_center_lat, bbox_center_lon, station_lat, station_lon)

                        if station_distance < min_distance:
                            nearest_station = row
                            min_distance = station_distance

                    # 更新dijkstra_pois字典
                    dijkstra_pois[(bbox_center_lat, bbox_center_lon)] = (nearest_station['Latitude'],
                                                                         nearest_station['Longitude'],
                                                                         nearest_station['Power'])

    # 添加选取的充电站坐标和power信息
    for bbox_center, (station_lat, station_lon, station_power) in dijkstra_pois.items():
        folium.Marker(location=[station_lat, station_lon],
                      popup=f'Latitude: {station_lat}<br>Longitude: {station_lon}<br>Power: {station_power}',
                      icon=folium.Icon(color='green')).add_to(m)

    # 保存地图为HTML文件
    m.save('charging_stations_map.html')



#################################################################

def dijkstra_edges():
    pois_df = pd.read_csv('dijkstra_pois.csv')
    # Define the number of points
    num_points = len(pois_df)
    # Create an empty matrix to store weights (initialize with infinity)
    weights = np.full((num_points, num_points), np.inf)
    # Loop through all pairs of points to calculate weights
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue  # Skip same point to same point
            point_A = pois_df.iloc[i]
            point_B = pois_df.iloc[j]
            x1, y1, c1, p1 = point_A['Latitude'], point_A['Longitude'], point_A['Elevation'], point_A['Power']
            x2, y2, c2, p2 = point_B['Latitude'], point_B['Longitude'], point_B['Elevation'], point_B['Power']

            # Calculate consumption and typical_duration
            consumption, typical_duration, _ = consumption_duration(x1, y1, c1, x2, y2, c2, mass, g, c_r, rho, A_front, c_d, a, eta_m, eta_battery)

            # Calculate charging time based on consumption
            charging_time = consumption / p2 if p2 > 0 else 0

            # Calculate total edge weight (driving time + charging time)
            total_weight = typical_duration + charging_time

            # If driving time exceeds 4.5 hours, set weight to infinity
            if typical_duration > 4.5 * 3600:
                total_weight = np.inf

            # Assign the weight to the matrix
            weights[i, j] = total_weight

    # Save the weights matrix to dijkstra_edges.csv
    pd.DataFrame(weights).to_csv('dijkstra_edges.csv', index=False)

    # Create a Folium map for visualization
    m = folium.Map(location=[pois_df['Latitude'].mean(), pois_df['Longitude'].mean()], zoom_start=10)

    # Loop through the weights matrix to add edges to the map
    for i in range(num_points):
        for j in range(num_points):
            weight = weights[i, j]
            if weight != np.inf:
                point_A = (pois_df.iloc[i]['Latitude'], pois_df.iloc[i]['Longitude'])
                point_B = (pois_df.iloc[j]['Latitude'], pois_df.iloc[j]['Longitude'])
                folium.PolyLine([point_A, point_B], color='blue', weight=1).add_to(m)

    # Save the map to a file
    m.save('dijkstra_map.html')



dijkstra_pois()









































