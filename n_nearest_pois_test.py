from environment import rp_env 
from nearest_location import nearest_location
from consumption_duration import consumption_duration
from consumption_duration import haversine
import pandas as pd

class way():
    def __init__(self):
        # initialization
        self.x_source = 52.66181  # source
        self.y_source = 13.38251
        self.c_source = 47
        self.x_target = 51.772324  # target
        self.y_target = 12.402652
        self.c_target = 88
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

    def info_way( self, x_current, y_current):

        






next_node = 6

n_ch =6
n_p = 4
n_pois = n_ch + n_p
file_path_ch = 'cs_combo_bbox.csv'
file_path_p = 'parking_bbox.csv'
x_current, y_current = 49.403861,9.390352
x_target, y_target = 51.772324, 12.402652
node_current = 3
self.data_ch = pd.read_csv(file_path_ch)
self.data_p = pd.read_csv(file_path_p)





# Obtain the altitude and/or power of current location
if node_current in range(0, n_ch):  # charging station
    c_current, power_current = cs_elevation_power(x_current, y_current)
else:  # parking lots
    c_current = p_elevation(x_current, y_current)

poi_files = [file_path_ch, file_path_p]
n_values = [n_ch, n_p]
nearest_n = []
for poi_file, n in zip(poi_files, n_values):
    nearest_poi = nearest_location(poi_file, x_current, y_current, n)
    for i in range(n):
        nearest_x = nearest_poi.loc[i, 'Latitude']
        nearest_y = nearest_poi.loc[i, 'Longitude']
        nearest_n.append([ nearest_x, nearest_y])
print(nearest_n)





coordinates_dict = {}
for i in range(n_pois):
    coordinates_dict[i] = nearest_n[i]
print(coordinates_dict)


# 根据next_node的值获取坐标
if next_node in range(n_ch):
    nearest_x, nearest_y = coordinates_dict[next_node]
    d_next = haversine(nearest_x, nearest_y, x_target, y_target)
    nearest_c, next_power = cs_elevation_power(nearest_x, nearest_y)
    consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, c_current,
                                                                        nearest_x, nearest_y,
                                                                        nearest_c, m, g,
                                                                        c_r, rho, A_front,
                                                                        c_d, a, eta_m, eta_battery)
else:
    nearest_x, nearest_y = coordinates_dict[next_node]
    d_next = haversine(nearest_x, nearest_y, x_target, y_target)
    nearest_c = p_elevation(nearest_x, nearest_y)
    consumption, typical_duration, length_meters = consumption_duration(x_current, y_current, c_current,
                                                                        nearest_x, nearest_y,
                                                                        nearest_c, m, g,
                                                                        c_r, rho, A_front,
                                                                        c_d, a, eta_m, eta_battery)

print(nearest_x, nearest_y)
print(d_next)
print("Length, speed, consumption", length_meters / 1000, "km", length_meters / typical_duration * 3.6, "km/h", consumption / length_meters * 100000, "kWh/100km\n")



