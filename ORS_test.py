import pandas as pd
import requests
import time

def get_elevation(api_key, latitude, longitude):
    base_url = "https://api.openrouteservice.org/elevation/point"
    params = {
        "api_key": api_key,
        "geometry": f"{latitude},{longitude}"
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        elevation = data["geometry"]["coordinates"][2]
        return elevation
    else:
        print(f"Failed to fetch elevation data. Status code: {response.status_code}")
        return None

# obtain an altitude of a POI
api_key = "5b3ce3597851110001cf624880a184fac65b416298dee8f52e43a0fe"
latitude = 49.0130
longitude = 8.4092
elevation = get_elevation(api_key, latitude, longitude)
print("Altitude：", elevation)


