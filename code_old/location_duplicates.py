# Date: 7/24/2023
# Author: Xubin Zhang
# Description: Find the duplicate coordinates in the two tables,
# and delete the duplicate coordinates in Table 1

import pandas as pd

#  Read the first table
file_path_1 = 'parking_location_noduplicate.csv'
df1 = pd.read_csv(file_path_1)
# Read the second table
file_path_2 = 'cs_location_bbox.csv'
df2 = pd.read_csv(file_path_2)

# Extract the coordinate from tables
coordinates_df1 = df1[['Latitude', 'Longitude']]
coordinates_df2 = df2[['Latitude', 'Longitude']]

#  Concatenate tables
all_coordinates = pd.concat([coordinates_df1, coordinates_df2])

# Find duplicate coordinates
duplicates = all_coordinates[all_coordinates.duplicated()]

if duplicates.empty:
    print("There are no duplicate coordinates in the two tables.")
else:
    print("Duplicate coordinates found in the first table:")
    print(duplicates)

# Remove duplicates from the first file
duplicates_array = duplicates.values.flatten()
df1_clean = df1[~df1[['Latitude', 'Longitude']].isin(duplicates_array).all(axis=1)]

# Save the cleaned first file without duplicate rows
output_file_path = 'parking_location_noduplicate_final.csv'
df1_clean.to_csv(output_file_path, index=False)

print("done")


