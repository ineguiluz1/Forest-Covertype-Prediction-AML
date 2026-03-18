import pandas as pd
from collections import Counter
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.preprocessing import LabelEncoder

DATA_DIR = 'data/interim/df_processed.parquet'
df = pd.read_parquet(DATA_DIR)

enc = LabelEncoder()
df['Aspect_Cardinal'] = enc.fit_transform(df['Aspect_Cardinal'])

print('Aspect_Cardinal mapping:', dict(zip(enc.classes_, enc.transform(enc.classes_))))

df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

df = df.dropna()

X = df.drop(columns=['Cover_Type', 'class_name'], errors='ignore')
y = df['Cover_Type']

print(f"Shape of the dataset: {Counter(y)}")

enn = EditedNearestNeighbours()

print("Applying Edited Nearest Neighbours...")

mode_enn = EditedNearestNeighbours(kind_sel = 'mode')
x_mode_enn, y_mode_enn = mode_enn.fit_resample(X,y)

df_mode_enn = pd.DataFrame(x_mode_enn, columns=X.columns)
df_mode_enn['Cover_Type'] = y_mode_enn

all_enn = EditedNearestNeighbours(kind_sel = 'all')
x_all_enn, y_all_enn = all_enn.fit_resample(X,y)

df_all_enn = pd.DataFrame(x_all_enn, columns=X.columns)
df_all_enn['Cover_Type'] = y_all_enn

df_mode_enn.to_parquet('data/processed/editedNN/editedNN_mode.parquet', index=False)
df_all_enn.to_parquet('data/processed/editedNN/editedNN_all.parquet', index=False)