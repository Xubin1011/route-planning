import pandas as pd
import heapq
from consumption_duration import haversine
# Sample data
latitudes = [40.7128, 34.0522, 51.5074, 48.8566, 37.7749]
longitudes = [-74.0060, -118.2437, -0.1278, 2.3522, -122.4194]
x_target, y_target = 40.7128, -74.0060  # Example target coordinates
n = 8  # Number of nearest locations to find

# Initialize closest_locations heap
closest_locations = []

# Iterate through all the locations
for lat, lon in zip(latitudes, longitudes):
    # Calculate the distance
    distance = haversine(x_target, y_target, lat, lon)
    # if distance < 25000:
    #     continue

    # dis_current = haversine(x_target, y_target, x_target, y_target)
    # dis_next = haversine(lat, lon, x_target, y_target)
    # if dis_current <= dis_next:  # only select pois close to the target
    #     continue

    # negate the distance to find the farthest distance,
    # closest_locations[0] is the farthest location now
    neg_distance = -distance

    # If the number of locations in the queue is less than n,
    # insert the current location
    if len(closest_locations) < n:
        heapq.heappush(closest_locations, (neg_distance, lat, lon))
    else:
        # find the farthest location in the current queue
        min_neg_distance, _, _ = closest_locations[0]

        # If the current location is closer, replace the farthest location
        if neg_distance > min_neg_distance:
            heapq.heappop(closest_locations)  # pop the farthest location
            heapq.heappush(closest_locations, (neg_distance, lat, lon))  # insert the closer location

# convert the distance back to positive values
closest_locations = pd.DataFrame(closest_locations, columns=["Neg_Distance", "Latitude", "Longitude"])
closest_locations["Distance"] = -closest_locations["Neg_Distance"]
closest_locations.drop(columns=["Neg_Distance"], inplace=True)

# Sort by distance in ascending order
closest_locations.sort_values(by="Distance", inplace=True)

# Extract information of the n nearest locations
nearest_locations = closest_locations.head(n).reset_index(drop=True)

# Print the result
print(nearest_locations)
print(len(nearest_locations))
nearest_locations_list = nearest_locations.values.tolist()
print(nearest_locations_list)

if len(nearest_locations_list) < n:
    for i in range(len(nearest_locations), n):
        nearest_locations_list.append ([1, 1, 1])
print(nearest_locations_list)