import pandas as pd
from calculate_functions import metod1, metod2, metod3
import os



def calculate_table(metod_dict: dict, files_name: list, parametrs: dict  ):
    if isinstance(files_name, str): 
        files_name = [files_name]
    column_list = ["Nuclei", "Cells", "Alive"]
    columns = ['File name']
    for metod_name in metod_dict:
        for value in column_list:
            columns.append(f"{metod_name}/{value}")
    

    table = pd.DataFrame(columns=columns)
    for file_path in files_name:
        row_toAdd = {"File name":os.path.basename(file_path)}

        for metod_name, metod in metod_dict.items():
            try:
                print(parametrs['Cell'],parametrs['Nuclei'])
                result = metod(img_path = file_path, cell_channel=parametrs['Cell'], nuclei_channel=parametrs['Nuclei'])
            except:
                result = None

            if result:
                for key, value in result.items():
                    if key == "%":
                        value = str(round(value,1))
                        value+="%"
                        key = "Alive"
                    if value == "-100%" or value == -100:
                        value = "-"
                    row_toAdd[f"{metod_name}/{key}"] = f"{value}"
            else:
                for i in column_list:

                    row_toAdd[f"{metod_name}/{i}"] = "-"
            row_to_add = pd.DataFrame([row_toAdd])
            table = pd.concat([table, row_to_add], ignore_index=True)
    return table
