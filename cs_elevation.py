# Date: 2023-07-18
# Author: Xubin Zhang
# Description:read the first rows_num rows of the charging_stations_location.excl,
# obtain the altitude through the openrouteservice api and insert the third column,
# delete the first rows_num rows of the original file


import pandas as pd
import requests
import time

def get_elevation(api_key, latitude, longitude):
    base_url = "https://api.openrouteservice.org/elevation/point"
    params = {
        "api_key": api_key,
        "geometry": f"{latitude},{longitude}",
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        elevation = data["geometry"]["coordinates"][2]
        return elevation
    else:
        print(f"Failed to fetch elevation data. Status code: {response.status_code}")
        return None

#input api key, file path, the number of rows
api_key = "5b3ce3597851110001cf624880a184fac65b416298dee8f52e43a0fe"
file_path = "cs_filtered_02.csv"
rows_num = 5

# read file
df = pd.read_csv(file_path)

#Get the latitude and longitude of the first rows_num rows,
#get the altitude information and save it in new column
for i, row in df.iloc[:rows_num].iterrows():
    df.at[i, "Elevation"] = get_elevation(api_key, row["Latitude"], row["Longitude"])
    time.sleep(1)  # Wait 1 second after each request

# Swap the "Power" column with the "Elevation" column
df = df[["Latitude", "Longitude", "Elevation", "Max_power"]]

# save to csv file
output_file_path = "cs_info.csv"
df.iloc[:rows_num].to_csv(output_file_path, index=False)

#Delete the first rows_num rows of the original table
df = df.iloc[rows_num:]
df.to_csv(file_path, index=False)

print("done")
