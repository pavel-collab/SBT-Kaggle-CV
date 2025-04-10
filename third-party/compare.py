from pathlib import Path
import os
import argparse
import numpy as np

import pandas as pd
import glob

accuracy = 1

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='set a path to directory with data')
parser.add_argument('-e', '--etalon', help='set a path to etalon csv file')
args = parser.parse_args()

data_path = Path(args.data)
etalon_filepath = Path(args.etalon)
assert(etalon_filepath.exists())
assert(data_path.exists())
assert(len(os.listdir(data_path.absolute())) != 0)

all_files = glob.glob(f"{data_path.absolute()}/" + "*.csv")
etalon_df = pd.read_csv(etalon_filepath.absolute())
etalon_data = etalon_df.iloc[:, 1:].round(accuracy).to_numpy()
summary_df = None

comparision_result = {}

# Чтение и суммирование данных из всех файлов
for filename in all_files:
    df = pd.read_csv(filename)
    # Удаляем первый столбец (индекс записи)
    data = df.iloc[:, 1:]  # Берем все столбцы, кроме первого
    
    res = np.isclose(data.round(accuracy).to_numpy(), etalon_data, 1e-2).sum()
    comparision_result[filename] = res
    
sorted_dict = dict(sorted(comparision_result.items(), key=lambda item: item[1], reverse=True))

for key, val in sorted_dict.items():
    print(f"Model: {key} val {val}")