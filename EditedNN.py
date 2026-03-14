import pandas as pd
from collections import Counter
from imblearn.under_sampling import EditedNearestNeighbours

DATA_DIR = 'data/raw/raw_data.csv'
df = pd.read_csv(DATA_DIR)

X = df.drop(columns=['Cover_Type', 'class_name'], errors='ignore')
y = df['Cover_Type']

print(f"Shape of the dataset: {Counter(y)}")

enn = EditedNearestNeighbours()

print("Applying Edited Nearest Neighbours...")

mode_enn = EditedNearestNeighbours(kind_sel = 'mode')
x_mode_enn, y_mode_enn = mode_enn.fit_resample(X,y)

x_mode_enn.to_parquet('data/processed/editedNN/editedNN_mode_X.parquet', index=False)
y_mode_enn.to_parquet('data/processed/editedNN/editedNN_mode_y.parquet')

all_enn = EditedNearestNeighbours(kind_sel = 'all')
x_all_enn, y_all_enn = all_enn.fit_resample(X,y)

x_all_enn.to_parquet('data/processed/editedNN/editedNN_all_X.parquet', index=False)
y_all_enn.to_parquet('data/processed/editedNN/editedNN_all_y.parquet')