# Date: 7/24/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...


import requests
import pandas as pd

def overpass_query(query):
    overpass_url = "https://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={"data": query})
    return response.json()

def get_node_data(node_id):
    query = f"""
    [out:json];
    node({node_id});
    out;
    """
    result = overpass_query(query)
    # obtaion lat and lon
    if "elements" in result and len(result["elements"]) > 0:
        node_data = result["elements"][0]
        node_coordinates = {"lat": node_data["lat"], "lon": node_data["lon"]}
        return node_coordinates
    return None
#         print(f"node {node_id} ：lat {node_coordinates['lat']}, lon {node_coordinates['lon']}")
#     else:
#         print(f"not found {node_id} ")
#
#print(get_node_data(5908313100))

def get_parking_data_in_bbox(bbox):
    query = f"""
    [out:json];
    (
        node["tourist_bus"="designated"]["amenity"="parking"]({bbox});
        way["tourist_bus"="designated"]["amenity"="parking"]({bbox});
    );
    out;
    """
    result = overpass_query(query)
    if "elements" in result:
        return result["elements"]
    return []

def get_coordinates(element):
    if element["type"] == "node":
        return get_node_data(element["id"])
    elif element["type"] == "way":
        first_node_id = element["nodes"][0]
        return get_node_data(first_node_id)
    return None

# example:Set the bounding box coordinates
bbox = "(49.013,8.409,52.525,13.369)"

# 获取符合条件的停车场数据
parking_data = get_parking_data_in_bbox(bbox)

# 提取停车场的经纬度信息
parking_coordinates = [get_coordinates(parking) for parking in parking_data]
parking_coordinates = [coord for coord in parking_coordinates if coord is not None]

# 转换为DataFrame并输出为CSV文件
df = pd.DataFrame(parking_coordinates, columns=["Latitude", "Longitude"])
df.to_csv("ebus_parking.csv", index=False)

print("OK")