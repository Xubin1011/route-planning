# Date: 2023-07-19
# Author: Xubin Zhang
# Description: Calculate unique types of sockets

#
# import pandas as pd
#
# # Read the data from the Excel file
# data = pd.read_csv('cs_filtered.csv')
#
# # Calculate unique values and their occurrences in 'Socket1', 'Socket2', 'Socket3', 'Socket4'
# socket_columns = ['Socket_1', 'Socket_2', 'Socket_3', 'Socket_4']
#
# unique_values = {}
# for column in socket_columns:
#     unique_values[column] = data[column].value_counts()
#
# # Print and save the results to 'types.csv'
# with open('types.csv', 'w',encoding='utf-8') as f:
#     for column, values_counts in unique_values.items():
#         f.write(f"Unique values and their occurrences in {column} column:\n")
#         f.write(f"{values_counts}\n\n")
#         print(f"Unique values and their occurrences in {column} column:")
#         print(values_counts)
#         print()



# Date: 2023-07-19
# Author: Xubin Zhang
# Description: the types of unneeded sockets will be replaced to 0

#
# import pandas as pd
#
# # Read the data from the Excel file
# data = pd.read_excel('Ladesaeulenregister-processed.xlsx')
#
# # Replace values in 'Socket_1' column with corresponding 'P1' values not being replaced
# # Socket_1 values to be replaced: 'DC Kupplung Combo', 'DC Kupplung Combo, DC CHAdeMO', 'AC Schuko',
# # 'DC Kupplung Combo, AC Schuko', 'DC Kupplung Combo, AC CEE 5 polig', 'AC CEE 5 polig', 'AC Schuko, AC CEE 5 polig'
# replace_values_socket1 = ['DC Kupplung Combo', 'DC Kupplung Combo, DC CHAdeMO', 'AC Schuko',
#                           'DC Kupplung Combo, AC Schuko', 'DC Kupplung Combo, AC CEE 5 polig', 'AC CEE 5 polig',
#                           'AC Schuko, AC CEE 5 polig', 'DC Kupplung Combo, AC CEE 3 polig']
# data['Socket_1'] = data['Socket_1'].replace(replace_values_socket1, 0)
#
# # Replace values in 'Socket_2' column with corresponding 'P2' values not being replaced
# # Socket_2 values to be replaced: 'DC Kupplung Combo', 'AC Schuko', 'DC CHAdeMO', 'AC CEE 5 polig',
# # 'AC Schuko, AC CEE 5 polig', 'DC Kupplung Combo, AC CEE 5 polig'
# replace_values_socket2 = ['DC Kupplung Combo', 'AC Schuko', 'DC CHAdeMO', 'AC CEE 5 polig',
#                           'AC Schuko, AC CEE 5 polig', 'DC Kupplung Combo, AC CEE 5 polig','DC Kupplung Combo, DC CHAdeMO']
# data['Socket_2'] = data['Socket_2'].replace(replace_values_socket2, 0)
#
# # Replace values in 'Socket_3' column with corresponding 'P3' values not being replaced
# # Socket_3 values to be replaced: 'DC Kupplung Combo', 'DC Kupplung Combo, DC CHAdeMO', 'AC Schuko', 'DC CHAdeMO'
# replace_values_socket3 = ['DC Kupplung Combo', 'DC Kupplung Combo, DC CHAdeMO', 'AC Schuko', 'DC CHAdeMO']
# data['Socket_3'] = data['Socket_3'].replace(replace_values_socket3, 0)
#
# # Replace values in 'Socket_4' column with corresponding 'P4' values not being replaced
# # Socket_4 values to be replaced: 'DC Kupplung Combo', 'AC Schuko', 'DC Kupplung Combo, DC CHAdeMO'
# replace_values_socket4 = ['DC Kupplung Combo', 'AC Schuko', 'DC Kupplung Combo, DC CHAdeMO']
# data['Socket_4'] = data['Socket_4'].replace(replace_values_socket4, 0)
#
# # Save the modified data to a new CSV file called 'cs_filtered.csv'
# data.to_csv('cs_filtered.csv', index=False)


# #test
# import pandas as pd
# data = pd.read_csv('cs_filtered.csv')
# row_10 = data.iloc[9]
# socket_2_value = row_10['Socket_2']
# print("Value in 'Socket_2' column (row 10):", socket_2_value)
# print("Type of value in 'Socket_2' column (row 10):", type(socket_2_value))


# Date: 2023-07-19
# Author: Xubin Zhang
# Description: Delete rows where type2 does not exist


# import pandas as pd
#
# # Read the data from the CSV file
# data = pd.read_csv('cs_filtered.csv')
#
# # Fill empty values with 0
# data.fillna(0, inplace=True)
#
# #if all values in 'Socket_1', 'Socket_2', 'Socket_3', 'Socket_4' columns are 0 and drop those rows
# socket_columns = ['Socket_1', 'Socket_2', 'Socket_3', 'Socket_4']
# #data = data[(data['Socket_1'] != 0) | (data['Socket_2'] != 0) | (data['Socket_3'] != 0) | (data['Socket_4'] != 0)]
# data = data[~((data[socket_columns] == '0') | (data[socket_columns] == 0)).all(axis=1)]
#
# # Save the modified DataFrame to 'cs_filtered_01.csv' file
# data.to_csv('cs_filtered_01.csv', index=False)
#
# # Output the number of rows in the table
# num_rows = data.shape[0]
# print("Number of rows in the table:", num_rows)

# # Date: 2023-07-19
# # Author: Xubin Zhang
# # Description:Compare the power of different sockets in a charging station,
# # and compare it with Nennleistung Ladeeinrichtung to select the maximum charging power
#
# import pandas as pd
#
# # Read the data from the CSV file
# data = pd.read_csv('cs_filtered_01.csv')
#
# # Replace 'P1', 'P2', 'P3', 'P4' values with 0 when 'Socket_1', 'Socket_2', 'Socket_3', 'Socket_4' values are 0
# data['P1'] = data['P1'].mask(data['Socket_1'].isin([0, '0']), 0).astype(int)
# data['P2'] = data['P2'].mask(data['Socket_2'].isin([0, '0']), 0).astype(int)
# data['P3'] = data['P3'].mask(data['Socket_3'].isin([0, '0']), 0).astype(int)
# data['P4'] = data['P4'].mask(data['Socket_4'].isin([0, '0']), 0).astype(int)
#
# # Compare values in 'P1', 'P2', 'P3', 'P4' columns and store the maximum value in a new column 'Max_socket_power'
# data['Max_socket_power'] = data[['P1', 'P2', 'P3', 'P4']].max(axis=1)
#
# # Compare 'Max_socket_power' column with values in 'Rated_output' column and store the minimum value in a new column 'Max_power'
# data['Max_power'] = data[['Max_socket_power', 'Rated_output']].min(axis=1)
#
# # Check if the 'Max_power' column is equal to the 'Rated_output' column and output the result
# is_equal = (data['Max_power'] == data['Rated_output']).all()
# print("Max_power column is equal to Rated_output column:", is_equal)
#
# # Save the modified DataFrame to 'cs_filtered_02.csv' file
# data.to_csv('cs_filtered_02.csv', index=False)

#check fast charging
import pandas as pd

data = pd.read_csv('cs_filtered_02.csv')

max_power_greater_than_22 = data[data['Max_power'] > 22]

if not max_power_greater_than_22.empty:
    print("Fast charging stations with Max_power greater than 22:")
    print(max_power_greater_than_22)
else:
    print("No fast charging stations found (Max_power > 22).")
