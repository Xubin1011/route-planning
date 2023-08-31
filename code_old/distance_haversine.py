# Date: 2023-07-18
# Author: Xubin Zhang
# Description: Calculate haversine distance between two points


import numpy as np
import math
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

# # test
# x1 = 49.013
# y1 = 8.409
# #x2, y2 = 49.054021, 8.535029
#
# x2 = 52.525 #berlin
# y2 = 13.369
#
# x1, y1 = 49.403861, 9.390352
# #
# x2, y2 = 51.557302, 12.9661
# #
# result = haversine(x1, y1, x2, y2)
# print(f"The distance is {result:.2f} meters")
