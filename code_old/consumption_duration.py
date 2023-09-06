# Date: 7/25/2023
# Author: Xubin Zhang
# Description: Calculate consumption between two POIs （in kWh）, get duration between two POIs （in s）

import numpy as np
from code_old.typical_route_here import get_typical_route_here
from code_old.distance_haversine import haversine


# def calculate_velocity(x1, y1, x2, y2):
#     # Get the typical duration between the two points
#     duration_seconds = get_typical_duration_here(x1, y1, x2, y2)
#
#     # Get the distance between the two points
#     distance_meters = haversine(x1, y1, x2, y2)
#
#     # Calculate the velocity
#     velocity = distance_meters / duration_seconds
#
#     return velocity, duration_seconds

# # Test
# x1, y1 = 49.013, 8.4092
# x2, y2 = 52.5253, 13.3693
#
# velocity = calculate_velocity(x1, y1, x2, y2)
#
# print("Duration of travel: ", get_typical_duration_here(x1, y1, x2, y2), " seconds")
# print("Distance of travel: ", get_length_here(x1, y1, x2, y2), " meters")
# print("Velocity: ", velocity, " meters per second")


#Description: alpha is the slope

def calculate_alpha(x1, y1, c1, x2, y2, c2):
    # Calculate the haversine distance
    distance_meters = haversine(x1, y1, x2, y2)

    print("Haversine Distance:", distance_meters, "m")
    # Calculate sinalpha based on c2-c1
    elevation_difference = c2 - c1
    slope = np.arctan(elevation_difference / distance_meters)  # (slope in radians) slope belongs to -pi/2 to pi/2
    sin_alpha = np.sin(slope)
    cos_alpha = np.cos(slope)

    # if elevation_difference > 0: #ascent
    #     slope = np.arctan(elevation_difference / distance_meters) #(slope in radians)
    #     sin_alpha = np.sin(slope)
    #     cos_alpha = np.cos(slope)
    #
    #     # sin_alpha = (c2 - c1) / distance_meters
    #     # cos_alpha = np.sqrt(1 - np.square(sin_alpha))
    # else:
    #     # descent is seen as no slope, treat the downhill road flat
    #     # driving at a constant speed,
    #     # regardless of Brake Energy Recuperation
    #     sin_alpha = 0
    #     cos_alpha = 1

    return sin_alpha, cos_alpha

# # Test
# x1, y1, c1 = 40.7128, -74.0060, 10
# x2, y2, c2 = 34.0522, -118.2437, 100
# result = calculate_alpha(x1, y1, c1, x2, y2, c2)
# print(result)


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

    # velocity, duration_seconds = calculate_velocity(x1, y1, x2, y2)


    typical_duration, length_meters, average_speed = get_typical_route_here(x1, y1, x2, y2)  # s, m, m/s

    # print(average_speed)
    # print(typical_duration)

    sin_alpha, cos_alpha = calculate_alpha(x1, y1, c1, x2, y2, c2)

    #print(sin_alpha,cos_alpha)

    # if average_speed > 27.8:
        # average_speed = 27.8
        # print("Speed limited")


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
            if power < -150000:  # 100kW
                power = -150000



    consumption = power * typical_duration / 3600 / 1000  #(in kWh)

    return consumption, typical_duration, length_meters

# test eCitaro 2 Türen
# x1, y1, c1 = 52.66181,13.38251, 47
# x2, y2, c2 = 51.772324,12.402652,88
# m = 13500 #(Leergewicht)
# g = 9.81
# rho = 1.225
# A_front = 10.03
# c_r = 0.01
# c_d = 0.7
# a = 0
# consumption, typical_duration, length_meters = consumption_duration(x1, y1, c1, x2, y2, c2, m, g, c_r, rho, A_front, c_d, a)
# print("Typical Duration:", typical_duration, "s")
# print("Consumption:", consumption, "kWh")
# print("Average comsuption:", consumption/length_meters*100000, "kWh/100km")


# x1, y1, c1 = 49.403861, 9.390352, 228
#
# x2, y2, c2 = 51.557302, 12.9661, 86
# sin_alpha, cos_alpha=calculate_alpha(x1, y1, c1, x2, y2, c2)
# print(sin_alpha)
# print(cos_alpha)

