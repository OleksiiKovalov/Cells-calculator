"""
This module defines the function to calculate a table based on certain methods applied to image files.
"""

# Standard library imports
import os
from typing import Dict, List

# Third-party imports
import pandas as pd


def _format_result_value(key, value):
    """
    Format result value based on key and value type.

    Args:
        key: The metric key.
        value: The value to format.

    Returns:
        Formatted value as string.
    """
    if value == -100:
        return "-"
    if key == "%":
        return f"{round(value, 1)}%"
    return str(value)


def calculate_table(
    model_dict: Dict[str, object],
    files_name: List[str],
    parameters: Dict[str, str],
) -> pd.DataFrame:
    """
    Calculate a table based on methods applied to image files.

    Args:
        model_dict: Dictionary with model names as keys and model objects as values.
        files_name: List of file paths (or single path string).
        parameters: Dict with 'Cell' and 'Nuclei' channel parameters.

    Returns:
        DataFrame with one row per file and columns for each model's metrics.
        
    Raises:
        ValueError: If files_name is empty or model_dict is empty.
    """
    # Convert single file name string to a list
    if isinstance(files_name, str):
        files_name = [files_name]
    
    if not files_name or not model_dict:
        raise ValueError("files_name and model_dict cannot be empty")

    # Define column structure
    column_list = ["Nuclei", "Cells", "Alive"]
    columns = ["File name"] + [
        f"{model_name}/{metric}"
        for model_name in model_dict
        for metric in column_list
    ]

    # Collect rows (more efficient than concatenating in loop)
    rows = []

    for file_path in files_name:
        row = {"File name": os.path.basename(file_path)}

        # Iterate through each method
        for model_name, model in model_dict.items():
            try:
                # Attempt to apply the method to the image file
                result = model.calculate(img_path=file_path, cell_channel=parameters['Cell'],\
                    nuclei_channel=parameters['Nuclei'])
            except:
                result = None

            if result:
                # If the method returns a result, add the values to the row dictionary
                for key, value in result.items():
                    row[f"{model_name}/{key}"] = _format_result_value(key, value)
            else:
                # If no result is returned, mark the row with "-"
                for i in column_list:
                    row[f"{model_name}/{i}"] = "-"

        # Add the completed row to the list
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
