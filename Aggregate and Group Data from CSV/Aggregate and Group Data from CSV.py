# Program: Aggregate and Group Data from CSV

import pandas as pd
df = pd.read_csv("data.csv")
print("Original Data:")
print(df)
grouped = df.groupby('Category').agg({
    'Sales': 'sum',       # Total sales per category
    'Quantity': 'mean',   # Average quantity per category
    'Profit': 'sum'       # Total profit per category
})
print("\nAggregated Data:")
print(grouped)
grouped.to_csv("aggregated_data.csv")
print("\nAggregated data has been saved to 'aggregated_data.csv'")
