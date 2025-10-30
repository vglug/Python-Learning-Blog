

import pandas as pd


file1 = 'data1.csv'
file2 = 'data2.csv'


df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)


merged_df = pd.merge(df1, df2, on='id', how='inner')

merged_df.to_csv('merged_output.csv', index=False)

print("✅ Files merged successfully! Output saved as 'merged_output.csv'")
