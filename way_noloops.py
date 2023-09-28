# Date: 2023-08-16
# Author: Xubin Zhang
# Description: Calculate the needed information between two pois
# Input: current location, current node, next node
# Output: Distance between next node and target, time and energy consumption between two points, location of next node, power if next node is a CS
# from nearest_location import nearest_location
from consumption_duration import consumption_duration
from consumption_duration import haversine
import pandas as pd
import heapq




class way():
    def __init__(self):
        # initialization
        self.n_ch = 6  # Number of the nearest charging stations
        self.n_p = 4  # Number of the nearest parking lots
        self.n_pois = 10

        self.x_source = 49.0130  # source
        self.y_source = 8.4093
        # self.c_source = 47
        self.x_target = 52.5253  # target
        self.y_target = 13.3694
        # self.c_target = 88
        self.m = 13500  # (Leergewicht) in kg
        self.g = 9.81
        self.rho = 1.225
        self.A_front = 10.03
        self.c_r = 0.01
        self.c_d = 0.7
        self.a = 0
        self.eta_m = 0.82
        self.eta_battery = 0.82

        self.file_path_ch = 'cs_combo_bbox.csv'
        self.file_path_p = 'parking_bbox.csv'
        self.data_ch = pd.read_csv("cs_combo_bbox.csv")
        self.data_p = pd.read_csv("parking_bbox.csv")

    def nearest_location(self, path, x1, y1, n):

        if path == self.file_path_ch:
            data = self.data_ch
        else:
            data = self.data_p

        latitudes = data["Latitude"]
        longitudes = data["Longitude"]

        # Priority queue to store information of the n nearest locations
        closest_locations = []

        # Iterate through all the locations
        for lat, lon in zip(latitudes, longitudes):
            # Calculate the distance
            distance = haversine(x1, y1, lat, lon)

            if distance < 25000:
                continue

            # negate the distance to find the farthest distance,
            # closest_locations[0] is the farthest location now
            neg_distance = -distance

            # If the number of locations in the queue is less than n,
            # insert the current location
            if len(closest_locations) < n:
                heapq.heappush(closest_locations, (neg_distance, lat, lon))
            else:
                # find the farthest location in the current queue
                min_neg_distance, _, _ = closest_locations[0]

                # If the current location is closer, replace the farthest location
                if neg_distance > min_neg_distance:
                    heapq.heappop(closest_locations)  # pop the farthest location
                    heapq.heappush(closest_locations, (neg_distance, lat, lon))  # insert the closer location

        # convert the distance back to positive values
        closest_locations = pd.DataFrame(closest_locations, columns=["Neg_Distance", "Latitude", "Longitude"])
        closest_locations["Distance"] = -closest_locations["Neg_Distance"]
        closest_locations.drop(columns=["Neg_Distance"], inplace=True)

        # Sort by distance in ascending order
        closest_locations.sort_values(by="Distance", inplace=True)

        # Extract information of the n nearest locations
        nearest_locations = closest_locations.head(n).reset_index(drop=True)
        return nearest_locations

    # obatin the elevation and power of a CS
    def cs_elevation_power(self, x, y):
        matching_row = self.data_ch[(self.data_ch["Latitude"] == x) & (self.data_ch["Longitude"] == y)]
        if not matching_row.empty:
            cs_elevation = matching_row["Elevation"].values[0]
            cs_power = matching_row["Power"].values[0]
        else:
            print("Current location not found in the dataset of cd")

        return (cs_elevation, cs_power)

    # obtain the elevation of a parking lat
    def p_elevation(self, x, y):
        matching_row = self.data_p[(self.data_p["Latitude"] == x) & (self.data_p["Longitude"] == y)]
        if not matching_row.empty:
            p_elevation = matching_row["Altitude"].values[0]
        else:
            print("Current location not found in the dataset of p")
        return (p_elevation)

    def info_way(self, node_current, x_current, y_current, node_next):
        # Obtain the altitude and/or power of current location
        if node_current in range(self.n_ch):  # charging station
            alti_current, power_current = self.cs_elevation_power(x_current, y_current)
        else:  # parking lots
            alti_current = self.p_elevation(x_current, y_current)

        # Obtain n_ch nearest charging stations and n_p nearest parking lots, saving in list nearest_n
        nearest_n = []
        poi_files = [self.file_path_ch, self.file_path_p]
        n_values = [self.n_ch, self.n_p]
        for poi_file, n in zip(poi_files, n_values):
            nearest_poi = self.nearest_location(poi_file, x_current, y_current, n)
            for i in range(n):
                next_x = nearest_poi.loc[i, 'Latitude']
                next_y = nearest_poi.loc[i, 'Longitude']
                nearest_n.append([next_x, next_y])
        # print("nearest n locations:", nearest_n)

        # Map the coordinates to the next location
        coordinates_dict = {}
        for i in range(self.n_pois):
            coordinates_dict[i] = nearest_n[i]
        # print(coordinates_dict)

        # Calculate the time and energy consumption between two points, the distance between next node and target
        next_x, next_y = coordinates_dict[node_next]
        d_next = haversine(next_x, next_y, self.x_target, self.y_target)

        if node_next in range(self.n_ch):
            alti_next, power_next = self.cs_elevation_power(next_x, next_y)
        else:
            alti_next = self.p_elevation(next_x, next_y)
            power_next = None
        consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, alti_current,
                                                                            next_x, next_y,
                                                                            alti_next, self.m, self.g,
                                                                            self.c_r, self.rho, self.A_front,
                                                                            self.c_d, self.a, self.eta_m,
                                                                            self.eta_battery)

        # delete current node
        # print(len(self.data_ch), len(self.data_p))
        if node_current in range(self.n_ch):
            mask = (self.data_ch['Latitude'] != x_current) | (self.data_ch['Longitude'] != y_current)
            self.data_ch = self.data_ch[mask]
        else:
            mask = (self.data_p['Latitude'] != x_current) | (self.data_p['Longitude'] != y_current)
            self.data_p = self.data_p[mask]

        # print(len(self.data_ch), len(self.data_p))

        return (next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters)

# #test
# x1 = 49.403861
# y1 = 9.390352
# node_c = 1
# node_next = 5
# myway = way()
# next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters = myway.info_way(node_c, x1, y1, node_next)
#
# print(next_x, next_y, d_next, power_next, consumption, typical_duration, length_meters)
# print("Length, average speed, average consumption", length_meters / 1000, "km", length_meters / typical_duration * 3.6, "km/h", consumption / length_meters * 100000, "kWh/100km\n")



