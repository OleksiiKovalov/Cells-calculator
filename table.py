import pandas as pd
from calculate_functions import metod1, metod2, metod3
import os
parametrs = {'Option 1': 'Channel 1',
                 'Option 2': 'Channel 2',
                 'Option 3': 'Channel 3'
                 
                 }
metods = {
        'metod1': metod1,
        'metod2': metod2,
        'metod3': metod3
    }


def calculate_table(metod_dict: dict, files_name: list, parametrs: dict  ):
    if isinstance(files_name, str): 
        files_name = [files_name]
    columns = ['file_name'].extend([metod_name for metod_name in metod_dict])
    table = pd.DataFrame(columns=columns)
    for file_path in files_name:
        row_toAdd = {"file_name":os.path.basename(file_path)}
        for metod_name, metod in metod_dict.items():
            result = metod(file_path, parametrs)
            if result:
                row_toAdd[metod_name] = f"{result['alive']*100}%/{result['dead']*100}%"
            else:
                row_toAdd[metod_name] = "-"
        table = table.append(row_toAdd, ignore_index = True)
    return table
