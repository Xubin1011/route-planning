import pandas as pd

def check_locations_exist(file1, file2):
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_excel(file2)

    # Create a set of tuples containing (Latitude, Longitude) from df2
    locations_set = set(tuple(x) for x in df2[['Latitude', 'Longitude']].values)

    # Initialize a counter for non-existing locations
    non_existing_count = 0

    # Check if each location in df1 exists in df2
    for index, row in df1.iterrows():
        latitude = row['Latitude']
        longitude = row['Longitude']

        if (latitude, longitude) not in locations_set:
            non_existing_count += 1

    print(f"The number of rows with locations that do not exist in table 2 is: {non_existing_count}")

    # if (latitude, longitude) in locations_set:
    #         print(f"Location ({latitude}, {longitude}) exists in both tables.")
    #     else:
    #         print(f"Location ({latitude}, {longitude}) does not exist in table 2.")

    return None


file1 = 'cs_data.csv'
file2 = 'Ladesaeulenregister-processed.xlsx'
check_locations_exist(file1, file2)