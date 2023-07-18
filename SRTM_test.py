import requests

def get_elevation(api_key, latitude, longitude, dem_type='=NASADEM'):
    base_url = 'https://portal.opentopography.org/API/globaldem'

    # Construct API request parameters
    params = {
        'demtype': dem_type,
        'south': latitude,
        'north': latitude,
        'west': longitude,
        'east': longitude,
        'outputFormat': 'json',
        'apiKey': api_key,
    }

    try:
        response = requests.get(base_url, params=params)
        response_json = response.json()

        if 'error' in response_json:
            print(f"Error: {response_json['error']['message']}")
            return None

        elevation = response_json['query']['dimensions']['elevation']
        return elevation

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

# API Key obtained after registration
api_key = '414d9bd2ce40f683703f7506709e1692'

# Coordinates of the point of interest
latitude = 49.013
longitude = 8.409

# Get the elevation for the point of interest
elevation = get_elevation(api_key, latitude, longitude)

if elevation is not None:
    print(f"The elevation at ({latitude}, {longitude}) is: {elevation} meters.")
else:
    print("Failed to retrieve elevation data.")
