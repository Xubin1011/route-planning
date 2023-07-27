# Date: 7/27/2023
# Author: Xubin Zhang
# Description: This file contains the implementation of...
#input： latitude and longitude of two pois
#output: parking_data.csv, include the latitude, longitude and elevation of all locations that has benn used.
# parking_bbox.csv, include the latitude, longitude and elevation of locations within bbox


import requests
import pandas as pd
import time
import os


api_key = "5b3ce3597851110001cf624880a184fac65b416298dee8f52e43a0fe"
rows_num = 5
source_lat, source_lon, target_lat, target_lon = 49.0130, 8.4093, 52.5253, 13.3694

def bounding_box(source_lat, source_lon, target_lat, target_lon):
    # Calculate the North latitude, West longitude, South latitude, and East longitude
    south_lat = min(source_lat, target_lat)
    west_lon = min(source_lon, target_lon)
    north_lat = max(source_lat, target_lat)
    east_lon = max(source_lon, target_lon)

    return south_lat, west_lon, north_lat, east_lon


#get locations within bbox and store in parking_bbox_tem.csv
def overpass_query(query):
    overpass_url = "https://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={"data": query})
    return response.json()

def get_parking_rest_area_services_data(bbox):
    query = f"""
    [out:json];
    (
        node["amenity"="parking"]["access"="yes"]{bbox};
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

    # Create DataFrame to store the locations
    parking_bbox_tem = pd.DataFrame(data, columns=["Latitude", "Longitude"])

    # Save to CSV
    parking_bbox_tem.to_csv("parking_bbox_tem.csv", index=False)

    return None


#Get elevation for new locations, and add to parking_bbox.csv, parking_data.csv,
#delete the first rows_num rows of parking_bbox_tem.csv
def parking_bbox_tem_altitude():
    # read file
    df_tem = pd.read_csv("parking_bbox_tem.csv")

    # Get the latitude and longitude of the first rows_num rows,
    # get the altitude information and save it in the third column
    for i, row in df_tem.iloc[:rows_num].iterrows():
        df_tem.at[i, "Altitude"] = get_elevation(row["Latitude"], row["Longitude"])
        time.sleep(1)  # Wait 1 second after each request

    # Read the existing 'parking_bbox.csv' file
    if os.path.exists('parking_bbox.csv'):
        df_bbox = pd.read_csv("parking_bbox.csv")
    else:
        df_bbox = pd.DataFrame(columns=['Latitude', 'Longitude', 'Altitude'])

    # Read the existing 'parking_data.csv' file
    if os.path.exists('parking_data.csv'):
        df_data = pd.read_csv("parking_data.csv")
    else:
        df_data = pd.DataFrame(columns=['Latitude', 'Longitude', 'Altitude'])

    # Append the updated rows_num data from df_tem to df_bbox
    df_bbox = df_bbox.append(df_tem.iloc[:rows_num], ignore_index=True)
    df_data = df_data.append(df_tem.iloc[:rows_num], ignore_index=True)

    # Save to 'parking_data.csv' file
    df_bbox.to_csv("parking_bbox.csv", index=False)
    df_data.to_csv("parking_data.csv", index=False)

    # Delete the first rows_num rows of the original table
    df_tem = df_tem .iloc[rows_num:]
    df_tem.to_csv("parking_bbox_tem.csv", index=False)

    return None


def compare_and_update_parking_data():
    # Check if parking_data.csv file exists
    if os.path.exists('parking_data.csv'):
        # Read the parking_data.csv file
        parking_data = pd.read_csv('parking_data.csv')
    else:
        # If the file does not exist, create a new DataFrame and add the header
        parking_data = pd.DataFrame(columns=['Latitude', 'Longitude', 'Altitude'])
        parking_data.to_csv('parking_data.csv', index=False)

    # read parking_bbox_tem.csv
    parking_bbox_tem = pd.read_csv('parking_bbox_tem.csv')

    # store the matched locations
    matching_rows_list = []

    # search the matched locations from parking_bbox_tem.csv and parking_data.csv
    for index, row in parking_bbox_tem.iterrows():
        latitude = row['Latitude']
        longitude = row['Longitude']

        # store the matched locations from parking_data.csv into matching_rows
        matching_rows = parking_data.loc[
            (parking_data['Latitude'] == latitude) & (parking_data['Longitude'] == longitude)]

        if not matching_rows.empty:
            # store in list
            matching_rows_list.append(matching_rows)

    if matching_rows_list:
        all_matching_rows = pd.concat(matching_rows_list)
        # store the matched locations in parking_bbox.csv
        all_matching_rows.to_csv('parking_bbox.csv', mode='a', index=False, header=False)
        # delete the matched locations in parking_bbox_tem.csv
        parking_bbox_tem.drop(all_matching_rows.index, inplace=True)
        # update parking_bbox_tem.csv
        parking_bbox_tem.to_csv('parking_bbox_tem.csv', index=False)

    return None



def get_elevation(latitude, longitude):
    base_url = "https://api.openrouteservice.org/elevation/point"
    params = {
        "api_key": api_key,
        "geometry": f"{longitude},{latitude}",
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        elevation = data["geometry"]["coordinates"][2]
        return elevation
    else:
        print(f"Failed to fetch elevation data. Status code: {response.status_code}")
        return None


def parking_data(source_lat, source_lon, target_lat, target_lon):

    bbox = bounding_box(source_lat, source_lon, target_lat, target_lon)

    #get locations within bbox and store in parking_bbox_tem.csv
    get_parking_rest_area_services_data(bbox)

    #get altitude from parking_data.csv
    #and delete matched location in parking_bbox_tem.csv
    #store the matched locations in parking_bbox.csv with altitude
    compare_and_update_parking_data()
    # # Get elevation for new locations, and add to parking_bbox.csv, parking_data.csv,
    # # delete the first rows_num rows of parking_bbox_tem.csv
    # parking_bbox_tem_altitude()

    return None



# Check if parking_bbox_tem.csv file exists
if os.path.exists('parking_bbox_tem.csv'):
    # Read the parking_bbox_tem.csv file into a DataFrame
    parking_bbox_tem_df = pd.read_csv('parking_bbox_tem.csv')

    # Check if the DataFrame is not empty
    if not parking_bbox_tem_df.empty:
        # If there are values in the DataFrame, get altitude
        parking_bbox_tem_altitude()
        # Read the parking_bbox_tem.csv file into a DataFrame
        parking_bbox_tem_df_1 = pd.read_csv('parking_bbox_tem.csv')
        # Check if the DataFrame is empty
        if parking_bbox_tem_df_1.empty:
            print("All altitudes have been obtained")
            os.remove('parking_bbox_tem.csv')
            print("The parking_bbox_tem.csv file is empty. Deleted the file.")
    else:
        # If the DataFrame is empty, delete the file
        os.remove('parking_bbox_tem.csv')
        print("The parking_bbox_tem.csv file is empty. Deleted the file.")
else:
    # If the file does not exist,get new location within bbox
    parking_data(source_lat, source_lon, target_lat, target_lon)

print("done")









