# Task #100: Basic Data Cleaning using pandas
import pandas as pd

# Read dataset (you can replace this with your CSV path)
data = pd.read_csv("data.csv")

# Show first few rows
print("Before Cleaning:\n", data.head())

# Drop rows with missing values
data_cleaned = data.dropna()

# Rename columns (example)
data_cleaned = data_cleaned.rename(columns={'OldColumnName': 'NewColumnName'})

# Reset index
data_cleaned.reset_index(drop=True, inplace=True)

# Save cleaned data
data_cleaned.to_csv("cleaned_data.csv", index=False)

print("\nAfter Cleaning:\n", data_cleaned.head())
print("\n✅ Data cleaned and saved as 'cleaned_data.csv'")
