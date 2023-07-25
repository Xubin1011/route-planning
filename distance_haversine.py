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
# x1 = 45.7597
# y1 = 4.8422
# x2 = 48.8567
# y2 = 2.3508
#
# result = haversine(x1, y1, x2, y2)
# print(f"The distance is {result:.2f} meters")
