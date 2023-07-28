# Date: 7/27/2023
# Author: Xubin Zhang
# Description: Clean data, extract the needed data
# (latitude, longitude, max_charging_power) from Ladesaeulenregister-processed.xlsx
# (Die Liste beinhaltet die Ladeeinrichtungen aller Betreiberinnen und Betreiber,
# die das Anzeigeverfahren der Bundesnetzagentur vollständig abgeschlossen und einer Veröffentlichung
# im Internet zugestimmt haben.)
#input: Ladesaeulenregister-processed.xlsx: latitude, longitude, all types of sockets
#output: cs_clean.csv: latitude, longitude, power (the charging power available for eCitaro)
# cs_data.csv: latitude, longitude, power and altitude.
# types.txt : the types of charging station in a table

import pandas as pd

#file_path_check_types = "Ladesaeulenregister-processed.xlsx"  # Calculate unique types of sockets
file_path_check_types = "combo_kept.csv"  # Calculate unique types of sockets
types_replaced_0 = "cs_data.csv"   # Unneeded types of sockets will be replaced to 0
file_path_delete = "combo_kept.csv" #Delete rows that combo does not exist

# Calculate unique types of sockets, output types.txt

def check_types(file_path_check_types):

    # Read the data from the Excel/csv file
    data = pd.read_csv(file_path_check_types)
    #data = pd.read_excel(file_path_check_types)

    # Calculate unique values and their occurrences in 'Socket1', 'Socket2', 'Socket3', 'Socket4'
    socket_columns = ['Socket_1', 'Socket_2', 'Socket_3', 'Socket_4']

    unique_values = {}
    for column in socket_columns:
        unique_values[column] = data[column].value_counts()

    # Print and save the results to 'types.txt'
    with open('combo_kept.txt', 'w', encoding='utf-8') as f:
        for column, values_counts in unique_values.items():
            f.write(f"Unique values and their occurrences in {column} column:\n")
            f.write(f"{values_counts}\n\n")

    print("types checked")
    return None


# Unneeded types of sockets will be replaced by 0
# output type2_combo_kept.csv

def keep_type2_combo(types_replaced_0):
    # Read the data from the Excel file
    data = pd.read_excel(types_replaced_0)

    # keep Type 2 and combo in a table
    # Replace values in 'Socket_1' column
    replace_values_socket1 = ['AC Schuko','AC CEE 5 polig', 'AC Schuko, AC CEE 5 polig']
    data['Socket_1'] = data['Socket_1'].replace(replace_values_socket1, 0)
    # Replace values in 'Socket_2' column
    replace_values_socket2 = ['AC Schuko', 'DC CHAdeMO', 'AC CEE 5 polig','AC Schuko, AC CEE 5 polig']
    data['Socket_2'] = data['Socket_2'].replace(replace_values_socket2, 0)
    # Replace values in 'Socket_3' column
    replace_values_socket3 = ['AC Schuko', 'DC CHAdeMO']
    data['Socket_3'] = data['Socket_3'].replace(replace_values_socket3, 0)
    # Replace values in 'Socket_4' column
    replace_values_socket4 = ['AC Schuko']
    data['Socket_4'] = data['Socket_4'].replace(replace_values_socket4, 0)
    # Save the modified data to a new CSV file
    data.to_csv('type2_combo_kept.csv', index=False)

    # Save the modified data to a new CSV file
    data.to_csv('type2_combo_kept.csv', index=False)

    print("replaced by 0, done")

    return None



# Unneeded types of sockets will be replaced by 0
# output combo_kept.csv

def keep_combo(types_replaced_0):
    # Read the data from the Excel file
    data = pd.read_excel(types_replaced_0)

    # keep only combo in a table
    # Replace values in 'Socket_1' column
    replace_values_socket1 = ['AC Steckdose Typ 2', 'AC Kupplung Typ 2', 'AC Steckdose Typ 2, AC Schuko', 'AC Steckdose Typ 2, AC Kupplung Typ 2',
                              'AC Kupplung Typ 2, AC Schuko', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko', 'AC Steckdose Typ 2, AC CEE 5 polig',
                              'AC Steckdose Typ 2, AC CEE 3 polig', 'AC Kupplung Typ 2, DC CHAdeMO', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC CEE 5 polig',
                              'AC Kupplung Typ 2, AC CEE 5 polig','AC Steckdose Typ 2, DC CHAdeMO', 'AC Schuko', 'AC Steckdose Typ 2, AC Schuko, AC CEE 5 polig',
                              'AC Kupplung Typ 2, AC CEE 3 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, DC CHAdeMO', 'AC Steckdose Typ 2, CEE-Stecker',
                              'AC CEE 5 polig','AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, AC CEE 3 polig; AC CEE 5 polig', 'AC Kupplung Typ 2, Adapter Typ1  Auto auf Typ2 Fahrzeugkupplung',
                              'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, DC CHAdeMO', 'AC Schuko, AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, AC CEE 5 polig',
                              'AC Steckdose Typ 2, AC Schuko, DC CHAdeMO']
    data['Socket_1'] = data['Socket_1'].replace(replace_values_socket1, 0)
    # Replace values in 'Socket_2' column
    replace_values_socket2 = ['AC Steckdose Typ 2', 'AC Steckdose Typ 2, AC Schuko', 'AC Kupplung Typ 2', 'AC Steckdose Typ 2, AC Kupplung Typ 2', 'AC Kupplung Typ 2, DC CHAdeMO',
                              'AC Schuko', 'AC Steckdose Typ 2, AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, DC CHAdeMO',
                              'AC Steckdose Typ 2, AC CEE 3 polig', 'AC Kupplung Typ 2, AC Schuko', 'DC CHAdeMO', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko',
                              'AC Steckdose Typ 2, AC Kupplung Typ 2, AC CEE 5 polig', 'AC Kupplung Typ 2, AC CEE 3 polig', 'AC Steckdose Typ 2, DC CHAdeMO',
                              'AC Kupplung Typ 2, AC CEE 5 polig', 'AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, AC CEE 3 polig; AC CEE 5 polig',
                              'AC Schuko, AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC / CEE']
    data['Socket_2'] = data['Socket_2'].replace(replace_values_socket2, 0)
    # Replace values in 'Socket_3' column
    replace_values_socket3 = ['AC Steckdose Typ 2', 'AC Kupplung Typ 2', 'AC Kupplung Typ 2, DC CHAdeMO', 'AC Steckdose Typ 2, AC Kupplung Typ 2', 'AC Steckdose Typ 2, AC Schuko',
                              'AC Steckdose Typ 2, AC Kupplung Typ 2, DC CHAdeMO', 'AC Schuko', 'DC CHAdeMO', 'AC Steckdose Typ 2, AC CEE 3 polig', 'AC Steckdose Typ 2, AC CEE 5 polig',
                              'AC Kupplung Typ 2, AC CEE 5 polig', 'AC Steckdose Typ 2, DC CHAdeMO', 'AC Kupplung Typ 2, AC Schuko', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko']
    data['Socket_3'] = data['Socket_3'].replace(replace_values_socket3, 0)
    # Replace values in 'Socket_4' column
    replace_values_socket4 = ['AC Steckdose Typ 2', 'AC Kupplung Typ 2', 'AC Schuko', 'AC Steckdose Typ 2, AC Kupplung Typ 2',
                              'AC Steckdose Typ 2, AC Schuko', 'AC Kupplung Typ 2, AC CEE 5 polig', 'AC Steckdose Typ 2, AC Kupplung Typ 2, AC Schuko, AC CEE 3 polig', 'AC Steckdose Typ 2, AC CEE 5 polig']
    data['Socket_4'] = data['Socket_4'].replace(replace_values_socket4, 0)
    # Save the modified data to a new CSV file
    data.to_csv('combo_kept.csv', index=False)

    print("replaced by 0, done")

    return None


# Delete rows that combo does not exist
# Delete unneeded power, replace 'P1', 'P2', 'P3', 'P4' values with 0 when Socket values are 0

def delete_unneeded_rows(file_path_delete):
    # Read the data from the CSV file
    data = pd.read_csv(file_path_delete)

    # Fill empty values with 0
    data.fillna(0, inplace=True)

    # Replace 'P1', 'P2', 'P3', 'P4' values with 0 when Socket values are 0
    data['P1'] = data['P1'].mask(data['Socket_1'].isin([0, '0']), 0).astype(int)
    data['P2'] = data['P2'].mask(data['Socket_2'].isin([0, '0']), 0).astype(int)
    data['P3'] = data['P3'].mask(data['Socket_3'].isin([0, '0']), 0).astype(int)
    data['P4'] = data['P4'].mask(data['Socket_4'].isin([0, '0']), 0).astype(int)

    # Compare power column with Rated_output column and select max charging power
    data['Max_socket_power'] = data[['P1', 'P2', 'P3', 'P4']].max(axis=1)


    # #if all values in Socket columns are 0 or '0', drop those rows
    # socket_columns = ['Socket_1', 'Socket_2', 'Socket_3', 'Socket_4']
    # #data = data[(data['Socket_1'] != 0) | (data['Socket_2'] != 0) | (data['Socket_3'] != 0) | (data['Socket_4'] != 0)]
    # data = data[~((data[socket_columns] == '0') | (data[socket_columns] == 0)).all(axis=1)]


    # Save the modified DataFrame to 'cs_filtered_01.csv' file
    data.to_csv('cs_filtered_03.csv', index=False)

    # Output the number of rows in the table
    num_rows = data.shape[0]
    print("Number of cs with combo:", num_rows)

    return None

delete_unneeded_rows(file_path_delete)




