from pathlib import Path
import os
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='set a path to directory with data')
args = parser.parse_args()

data_path = Path(args.data)
assert(data_path.exists())
assert(len(os.listdir(data_path.absolute())) != 0)

all_files = glob.glob(data_path.absolute() + "*.csv")

# Чтение и суммирование данных из всех файлов
for filename in all_files:
    df = pd.read_csv(filename)
    
    # Удаляем первый столбец (индекс записи)
    data = df.iloc[:, 1:]  # Берем все столбцы, кроме первого
    
    if summary_df.empty:
        summary_df = data
    else:
        summary_df += data  # Суммируем данные

# Создание heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(summary_df, annot=True, fmt=".2f", cmap='viridis')

plt.title('Heatmap of Summed Values from CSV Files')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()