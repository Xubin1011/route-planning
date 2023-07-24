# Date: 2023-07-24
# Author: Xubin Zhang
# Description: Check duplicate rows in csv file





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





import pandas as pd

# Read the CSV file
df = pd.read_csv('cs_filtered_02_noduplicate_02.csv')

# Find duplicate rows based on 'Latitude' and 'Longitude'
duplicate_coords = df[df.duplicated(['Latitude', 'Longitude'], keep=False)]

# Output the duplicate rows
if duplicate_coords.shape[0] == 0:
    print("No duplicate rows found.")
else:
    print("Duplicate rows found:")
    print(duplicate_coords)





#Same 'Latitude' and 'Longitude', the row with the maximum 'Max_power' values will be saved.
#Duplicate rows will be removed

# import pandas as pd
#
# # Read the CSV file
# df = pd.read_csv('cs_filtered_02_noduplicate_02.csv')
#
# # Find duplicate rows based on 'Latitude' and 'Longitude'
# duplicate_coords = df[df.duplicated(['Latitude', 'Longitude'], keep=False)]
#
# # Find the row with the maximum 'Max_power' value among duplicates
# max_power_row = duplicate_coords.loc[duplicate_coords.groupby(['Latitude', 'Longitude'])['Max_power'].idxmax()]
#
# # Concatenate the row with the maximum 'Max_power' value and rows with unique coordinates
# result_df = pd.concat([df.drop(duplicate_coords.index), max_power_row])
#
# # Save the final result to a new CSV file
# result_df.to_csv('cs_filtered_02_noduplicate_final.csv', index=False)
# print("done")





