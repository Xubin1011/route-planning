# Date: 7/18/2023
# Author: Xubin Zhang
# Description: invalid, the tags of charging stations are incomplete, chaotic

import requests
import pandas as pd


def get_charging_stations(south_lat, west_lon, north_lat, east_lon):
    overpass_url = "https://overpass-api.de/api/interpreter"
    bbox = f"({south_lat},{west_lon},{north_lat},{east_lon})"
    query = f"""
    [out:json];
    (
      node["amenity"="charging_station"]{bbox};
    );
    out;
    """
    response = requests.get(overpass_url, params={'data': query})
    return response.json()


def filter_charging_stations(data):
    filtered_stations = []
    for element in data['elements']:
        tags = element.get('tags', {})
        access = tags.get('access', None)
        socket_type = tags.get('socket:type', None)
        if access == 'yes' and socket_type in ['type2', 'type2_cable', 'type2_combo', 'chademo', 'tesla_destination']:
            filtered_stations.append(element)
    return filtered_stations


def extract_data(filtered_stations):
    charging_data = []
    for element in filtered_stations:
        lat = element['lat']
        lon = element['lon']
        elevation = element.get('elevation', None)

        # Extracting maximum power information from the charging_station:output tag
        max_output = element.get('tags', {}).get('charging_station:output', None)
        max_power = max_output.split('=')[1].strip() if max_output and 'kW' in max_output else None

        charging_data.append((lat, lon, elevation, max_power))
    return charging_data


south_lat, west_lon = 49.013, 8.409
north_lat, east_lon = 52.525, 13.369

response_data = get_charging_stations(south_lat, west_lon, north_lat, east_lon)
filtered_stations = filter_charging_stations(response_data)
charging_data = extract_data(filtered_stations)

df = pd.DataFrame(charging_data, columns=['Latitude', 'Longitude', 'Elevation', 'Maximum Power'])

# Output DataFrame to CSV file
df.to_csv('charging_stations_osm.csv', index=False)

