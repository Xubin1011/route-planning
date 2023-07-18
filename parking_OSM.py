import requests
import pandas as pd

def overpass_query(query):
    overpass_url = "https://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={"data": query})
    return response.json()

def get_parking_rest_area_services_data(bbox):
    query = f"""
    [out:json];
    (
        node["amenity"="parking"]["access"~"^(yes|permissive)$"]{bbox};
        node["highway"="rest_area"]{bbox};
        node["highway"="services"]{bbox};
    );
    out;
    """
    response_json = overpass_query(query)
    data = []
    for element in response_json["elements"]:
        if element["type"] == "node":
            lat = element["lat"]
            lon = element["lon"]
            data.append((lat, lon))
    return data

# Set the bounding box coordinates
bbox = "(49.013,8.409,52.525,13.369)"

data = get_parking_rest_area_services_data(bbox)

# Create DataFrame
df = pd.DataFrame(data, columns=["Latitude", "Longitude"])

# Save to CSV
df.to_csv("parking_rest_area_services_filtered.csv", index=False)
