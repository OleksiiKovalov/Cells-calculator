"""
This module defines the function to calculate a table based on certain methods applied to image files.
"""
import os
import pandas as pd


def calculate_table(model_dict: dict, files_name: list, parametrs: dict):
    """
    Calculate a table based on methods applied to image files.

    Args:
    model_dict (dict): Dictionary containing method names as keys and corresponding functions as values.
    files_name (list): List of file names or a single file name.
    parametrs (dict): Dictionary containing parameters for the methods.

    Returns:
    DataFrame: A pandas DataFrame representing the calculated table.
    """
    # Convert single file name string to a list
    if isinstance(files_name, str): 
        files_name = [files_name]

    # Define column names for the table
    column_list = ["Nuclei", "Cells", "Alive"]
    columns = ['File name']
    for model_name in model_dict:
        for value in column_list:
            columns.append(f"{model_name}/{value}")

    # Create an empty DataFrame with the defined columns
    table = pd.DataFrame(columns=columns)

    # Iterate through each file
    for file_path in files_name:
        # Initialize a dictionary for the row to add to the table
        row_toAdd = {"File name": os.path.basename(file_path)}

        # Iterate through each method
        for model_name, model in model_dict.items():
            try:
                # Attempt to apply the method to the image file
                result = model.calculate(img_path=file_path, cell_channel=parametrs['Cell'],\
                    nuclei_channel=parametrs['Nuclei'])
            except:
                result = None

            if result:
                # If the method returns a result, add the values to the row dictionary
                for key, value in result.items():
                    if key == "%":
                        value = str(round(value, 1))
                        value += "%"
                        key = "Alive"
                    if value == "-100%" or value == -100:
                        value = "-"
                    row_toAdd[f"{model_name}/{key}"] = f"{value}"
            else:
                # If no result is returned, mark the row with "-"
                for i in column_list:
                    row_toAdd[f"{model_name}/{i}"] = "-"

            # Convert the row dictionary to a DataFrame and concatenate it with the main table
            row_to_add = pd.DataFrame([row_toAdd])
            table = pd.concat([table, row_to_add], ignore_index=True)
    return table
