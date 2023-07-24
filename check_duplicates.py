# Date: 2023-07-24
# Author: Xubin Zhang
# Description: Check duplicate rows in csv file



# Description: Check duplicate rows of parking

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('parking_location.csv')
#
# # Find duplicate rows based on 'Latitude' and 'Longitude'
# duplicate_coords = df[df.duplicated(['Latitude', 'Longitude'], keep=False)]
#
# # Output the duplicate rows
# if duplicate_coords.shape[0] == 0:
#     print("No duplicate rows found.")
# else:
#     print("Duplicate rows found:")
#     print(duplicate_coords)
#
# # Remove duplicate rows
# df.drop_duplicates(subset=['Latitude', 'Longitude'], inplace=True)
#
# # Save the DataFrame without duplicate rows to a new CSV file
# df.to_csv('parking_location_noduplicate.csv', index=False)
# print("done.")




# Description: Check duplicate rows of charging stations

# import pandas as pd
#
# # Read CSV file
# df = pd.read_csv('cs_filtered_02.csv')
#
# # Find duplicate rows
# duplicate_rows = df[df.duplicated()]
#
# if duplicate_rows.shape[0] == 0:
#     print("No duplicate rows found.")
# else:
#     print("Duplicate rows found:")
#     print(duplicate_rows)
#
# # Remove duplicate rows
# df.drop_duplicates(inplace=True)
#
# # Save the DataFrame back to the CSV file
# df.to_csv('cs_filtered_02_noduplicate.csv', index=False)
# print("done")




# Description: Extract needed info of CS

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('cs_filtered_02_noduplicate.csv')
#
# # Extract 'Latitude', 'Longitude', and 'Max_power' columns
# selected_columns = df[['Latitude', 'Longitude', 'Max_power']]
#
# # Save the selected columns to a new CSV file
# selected_columns.to_csv('cs_filtered_02_noduplicate_01.csv', index=False)





# Description: Check duplicate rows after extract

# import pandas as pd
#
# df = pd.read_csv('cs_filtered_02_noduplicate_01.csv')
# duplicate_rows = df[df.duplicated()]
# if duplicate_rows.shape[0] == 0:
#     print("No duplicate rows found.")
# else:
#     print("Duplicate rows found:")
#     print(duplicate_rows)
#
# # Remove duplicate rows
# df.drop_duplicates(inplace=True)
#
# df.to_csv('cs_filtered_02_noduplicate_02.csv', index=False)
# print("done")



# Description: Check duplicate based on 'Latitude' and 'Longitude'

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('cs_filtered_02_noduplicate_02.csv')
#
# # Find duplicate rows based on 'Latitude' and 'Longitude'
# duplicate_coords = df[df.duplicated(['Latitude', 'Longitude'], keep=False)]
#
# # Output the duplicate rows
# if duplicate_coords.shape[0] == 0:
#     print("No duplicate rows found.")
# else:
#     print("Duplicate rows found:")
#     print(duplicate_coords)





# Find same 'Latitude' and 'Longitude',
# save the row with the maximum 'Max_power' value

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('cs_filtered_02_noduplicate_02.csv')
#
# # Find duplicate rows based on 'Latitude' and 'Longitude'
# duplicate_coords = df[df.duplicated(['Latitude', 'Longitude'], keep=False)]
#
# # Find the row with the maximum 'Max_power' value among duplicates
# max_power_rows = duplicate_coords.loc[duplicate_coords.groupby(['Latitude', 'Longitude'])['Max_power'].idxmax()]
#
# # Drop duplicate rows and keep the row with the maximum 'Max_power' value
# df.drop_duplicates(subset=['Latitude', 'Longitude'], keep=False, inplace=True)
#
# # Concatenate the rows with the maximum 'Max_power' value and rows with unique coordinates
# result_df = pd.concat([df, max_power_rows])
#
# # Save the final result to a new CSV file
# result_df.to_csv('cs_filtered_02_noduplicate_final.csv', index=False)
# print("done")


#check again

import pandas as pd

# Read the CSV file
df = pd.read_csv('cs_filtered_02_noduplicate_final.csv')

# Find duplicate rows based on 'Latitude' and 'Longitude'
duplicate_coords = df[df.duplicated(['Latitude', 'Longitude'], keep=False)]

# Output the duplicate rows
if duplicate_coords.shape[0] == 0:
    print("No duplicate rows found.")
else:
    print("Duplicate rows found:")
    print(duplicate_coords)




