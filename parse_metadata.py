import pandas as pd

metadata_path = "data/archive(2)/HAM10000_metadata.csv"

df = pd.read_csv(metadata_path, sep = ",")
print(df)

classes = df.dx.unique()
print(classes)

