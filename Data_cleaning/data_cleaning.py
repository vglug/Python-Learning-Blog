import pandas as pd
data = pd.read_csv("data.csv")
print("Before Cleaning:\n", data.head())
data_cleaned = data.dropna()
data_cleaned = data_cleaned.rename(columns={'OldColumnName': 'NewColumnName'})
data_cleaned.reset_index(drop=True, inplace=True)
data_cleaned.to_csv("cleaned_data.csv", index=False)
print("\nAfter Cleaning:\n", data_cleaned.head())
print("\nData cleaned and saved as 'cleaned_data.csv'")
