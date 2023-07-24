# Date: 2023-07-18
# Author: Xubin Zhang
# Description: Extract latitude and longitude within the specified bounding box range
# from charging_stations_location.excel


import pandas as pd

# Define the bounding box format
bbox = "(49.013,8.409,52.525,13.369)"

# Extract latitude and longitude values from the bounding box format
bbox_values = bbox.strip('()').split(',')
south_lat, west_lon, north_lat, east_lon = map(float, bbox_values)

# Read data from excel file into a Pandas DataFrame
df = pd.read_csv('cs_filtered_02.csv')

# Filter the rows within the bounding box range
filtered_df = df[
    (df['Latitude'] >= south_lat) & (df['Latitude'] <= north_lat) &
    (df['Longitude'] >= west_lon) & (df['Longitude'] <= east_lon)
]

# Save the filtered result to a CSV file
filtered_df.to_csv('cs_info_bbox.csv', index=False)

print("done")

