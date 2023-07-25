# Date: 7/18/2023
# Author: Xubin Zhang
# Description:
# get_typical_duration_here: get typicalDuration between two POIs from Here API
# get_base_duration_here: get baseDuration between two POIs from Here API
#get_length_here: get length between two POIs from Here API


import requests
import time

def get_typical_duration_here(x1, y1, x2, y2):
    api_key = "_XecI_2z9_7QVDELBzW_dT8VeRjUW4uJtOkxpm4Qvrs"
    base_url = "https://router.hereapi.com/v8/routes"
    params = {
        "apiKey": api_key,
        "transportMode": "car",
        "origin": f"{x1},{y1}",
        "destination": f"{x2},{y2}",
        "return": "summary,typicalDuration",
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # delay of 1s after each request
    time.sleep(0.5)

    try:
        # Extract typicalDuration from response
        typical_duration = data["routes"][0]["sections"][0]["summary"]["typicalDuration"]

        # print("departure_time:", data["routes"][0]["sections"][0]["departure"]["time"])
        # print("Summary:", data["routes"][0]["sections"][0]["summary"])

        #return typical_duration
        return typical_duration
    except (KeyError, IndexError):
        # Return None if the required data is not available in the response
        return None, None

# # test
# x1, y1 = 49.013, 8.4092
# x2, y2 = 52.5253, 13.3693
#
# typical_duration = get_typical_duration_here(x1, y1, x2, y2)
# print("Typical Duration:", typical_duration, "seconds")
#


def get_base_duration_here(x1, y1, x2, y2):
    api_key = "_XecI_2z9_7QVDELBzW_dT8VeRjUW4uJtOkxpm4Qvrs"
    base_url = "https://router.hereapi.com/v8/routes"
    params = {
        "apiKey": api_key,
        "transportMode": "car",
        "origin": f"{x1},{y1}",
        "destination": f"{x2},{y2}",
        "return": "summary",
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # delay of 1s after each request
    time.sleep(0.5)

    try:
        # Extract baseDuration response

        base_duration = data["routes"][0]["sections"][0]["summary"]["baseDuration"]

        # print("Summary:", data["routes"][0]["sections"][0]["summary"])

        return base_duration
    except (KeyError, IndexError):
        # Return None if the required data is not available in the response
        return None, None


# # test
# x1, y1 = 49.013, 8.4092
# x2, y2 = 52.5253, 13.3693
# base_duration = get_base_duration_here(x1, y1, x2, y2)
# print("Base Duration:", base_duration, "seconds")



def get_length_here(x1, y1, x2, y2):
    api_key = "_XecI_2z9_7QVDELBzW_dT8VeRjUW4uJtOkxpm4Qvrs"
    base_url = "https://router.hereapi.com/v8/routes"
    params = {
        "apiKey": api_key,
        "transportMode": "car",
        "origin": f"{x1},{y1}",
        "destination": f"{x2},{y2}",
        "return": "summary",
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    time.sleep(0.5)
    try:
        # Extract length
        length_meters = data["routes"][0]["sections"][0]["summary"]["length"]

        return length_meters
    except (KeyError, IndexError):
        # Return None if the required data is not available in the response
        return None

# # test
# x1, y1 = 49.013, 8.4092
# x2, y2 = 52.5253, 13.3693
#
# length = get_length_here (x1, y1, x2, y2)
# print("Length:", length, "meters")
