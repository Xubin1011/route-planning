# Date: 2023-07-18
# Author: Xubin Zhang
# Description: This file contains the implementation of...


import csv
from distance_haversine import haversine

def read_csv(file_name):
    data = []
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            data.append([float(row[0]), float(row[1])])
    return data

def find_nearest_locations(current_lat, current_lon, locations_table, dmax, n=2):
    # Create a dictionary to store the mapping of locations and distances
    distances = {}
    for row in locations_table:
        lat, lon = row[0], row[1]
        distance = haversine(current_lat, current_lon, lat, lon)
        if distance <= dmax:
            distances[tuple(row)] = distance

    # Sort based on distances and take the first n locations
    nearest_locations = sorted(distances.items(), key=lambda x: x[1])[:n]

    return nearest_locations

# example:
current_latitude = 37.7749
current_longitude = -122.4194
dmax = 5000  # Assume dmax is 5000 meters

# Read the data from the CSV files
parking_data = read_csv("parking.csv")
cs_data = read_csv("cs.csv")

# Find the nearest parking locations and charging stations
nearest_parking = find_nearest_locations(current_latitude, current_longitude, parking_data, dmax, n=2)
nearest_cs = find_nearest_locations(current_latitude, current_longitude, cs_data, dmax, n=3)

print("Nearest parking locations:")
for location, distance in nearest_parking:
    print(f"Location: {location}, Distance: {distance} meters")

print("\nNearest charging station locations:")
for location, distance in nearest_cs:
    print(f"Location: {location}, Distance: {distance} meters")
