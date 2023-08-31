# Date: 7/18/2023
# Author: Xubin Zhang
# Description:
# get_typical_route_here: get typicalDuration, length, and calculate average speed between two POIs from Here API
# departure_time:current time (Default)
# routingMode: fast (Default) (Route calculation from start to destination optimized by travel time.)


import requests
import time

def get_typical_route_here(x1, y1, x2, y2):
    api_key = "_XecI_2z9_7QVDELBzW_dT8VeRjUW4uJtOkxpm4Qvrs"
    base_url = "https://router.hereapi.com/v8/routes"
    params = {
        "apiKey": api_key,
        "transportMode": "privateBus",
        "origin": f"{x1},{y1}",
        "destination": f"{x2},{y2}",
        "return": "summary,typicalDuration",
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # delay of 1s after each request
    time.sleep(1)

    try:
        # Extract typicalDuration from response
        typical_duration = data["routes"][0]["sections"][0]["summary"]["typicalDuration"]
        length_meters = data["routes"][0]["sections"][0]["summary"]["length"]
        #base_duration = data["routes"][0]["sections"][0]["summary"]["baseDuration"]

        average_speed = length_meters / typical_duration #(m/s)
        #average_speed = length_meters / base_duration

        print("Departure time:", data["routes"][0]["sections"][0]["departure"]["time"])
        print("Summary:", data["routes"][0]["sections"][0]["summary"])
        print("Average speed:", average_speed, "m/s", "=", average_speed * 3.6, "km/h")

        #return typical_duration
        return typical_duration, length_meters, average_speed  # s, m, m/s
    except (KeyError, IndexError):
        # Return None if the required data is not available in the response
        return None, None

# # test
# x1, y1 = 49.013, 8.4092
# #x2, y2 = 49.054021, 8.535029
# x2, y2 = 52.5253, 13.3693 #berlin

# x1, y1, x2, y2 = 52.096647, 7.228437, 51.28297056,8.873471783
#
#
# typical_duration, length_meters, average_speed = get_typical_route_here(x1, y1, x2, y2)
# print("Typical Duration:", typical_duration, "s")
# print("Length meters:", length_meters, "m")
# print("average speed:", average_speed, "m/s", "=", average_speed * 3.6, "km/h")



# def get_base_duration_here(x1, y1, x2, y2):
#     api_key = "_XecI_2z9_7QVDELBzW_dT8VeRjUW4uJtOkxpm4Qvrs"
#     base_url = "https://router.hereapi.com/v8/routes"
#     params = {
#         "apiKey": api_key,
#         "transportMode": "privateBus",
#         "origin": f"{x1},{y1}",
#         "destination": f"{x2},{y2}",
#         "return": "summary",
#     }
#
#     response = requests.get(base_url, params=params)
#     data = response.json()
#
#     # delay of 1s after each request
#     time.sleep(0.5)
#
#     try:
#         # Extract baseDuration response
#
#         base_duration = data["routes"][0]["sections"][0]["summary"]["baseDuration"]
#
#         # print("Summary:", data["routes"][0]["sections"][0]["summary"])
#
#         return base_duration
#     except (KeyError, IndexError):
#         # Return None if the required data is not available in the response
#         return None, None


# # test
# x1, y1 = 49.013, 8.4092
# x2, y2 = 52.5253, 13.3693
# base_duration = get_base_duration_here(x1, y1, x2, y2)
# print("Base Duration:", base_duration, "seconds")



