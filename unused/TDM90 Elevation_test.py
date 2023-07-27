import requests


def get_elevation(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if 'elevation' in data:
            elevation = data['elevation']
            return elevation
        else:
            print("Elevation data not found.")
    else:
        print("Failed to retrieve data.")

    return None


# 测试：纽约市的经纬度
latitude = 40.7128
longitude = -74.0060
elevation = get_elevation(latitude, longitude)

if elevation:
    print(f"The elevation at ({latitude}, {longitude}) is {elevation} meters.")
