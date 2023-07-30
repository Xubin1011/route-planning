import numpy as np
from typical_route_here import get_typical_route_here
from distance_haversine import haversine

#Description: alpha is the slope

def calculate_alpha(x1, y1, c1, x2, y2, c2):
    # Calculate the haversine distance
    distance_meters = haversine(x1, y1, x2, y2)

    print("Haversine Distance:", distance_meters, "m")
    # Calculate sinalpha based on c2-c1
    elevation_difference = c2 - c1
    if elevation_difference > 0: #ascent
        slope = np.arctan(elevation_difference / distance_meters)


        sin_alpha = np.sin(slope)
        print(sin_alpha)
        cos_alpha = np.cos(slope)
        print(cos_alpha)
        tan_alpha = np.tan(slope)
        print(tan_alpha)

        sin_theta = elevation_difference / distance_meters
        print(sin_theta)
        cos_theta = np.sqrt(1 - np.square(sin_alpha))
        print(cos_theta)
        tan_theta = sin_theta/cos_theta
        print(tan_theta)
    else:
        # descent is seen as no slope, treat the downhill road flat
        # driving at a constant speed,
        # regardless of Brake Energy Recuperation
        sin_alpha = 0
        cos_alpha = 1

    return sin_alpha, cos_alpha,tan_alpha, sin_theta, cos_theta, tan_theta

# Test
x1, y1, c1 = 40.7128, -74.0060, 10
x2, y2, c2 = 34.0522, -118.2437, 800
result = calculate_alpha(x1, y1, c1, x2, y2, c2)
print(result)