import pandas as pd

df = pd.read_csv("D:/frame_labels.csv")
print(df['label'].value_counts())
