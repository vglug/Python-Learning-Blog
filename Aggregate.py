import pandas as pd

# Read CSV file
file_path = input("Enter the CSV file path: ")
data = pd.read_csv(file_path)

print("\nFirst few rows of the dataset:")
print(data.head())

# Ask user for grouping and aggregation columns
group_col = input("\nEnter the column name to group by: ")
agg_col = input("Enter the column name to aggregate: ")

# Perform aggregation (sum, mean, count, etc.)
print("\nChoose aggregation method:")
print("1. Sum")
print("2. Mean")
print("3. Count")
print("4. Maximum")
print("5. Minimum")

choice = input("Enter your choice (1-5): ")

agg_method = {
    "1": "sum",
    "2": "mean",
    "3": "count",
    "4": "max",
    "5": "min"
}.get(choice, "sum")

# Group and aggregate
result = data.groupby(group_col)[agg_col].agg(agg_method).reset_index()

# Display results
print(f"\nGrouped and aggregated data ({agg_method} of '{agg_col}' by '{group_col}'):")
print(result)

# Optionally save output
save_choice = input("\nDo you want to save the result as a new CSV file? (yes/no): ").lower()
if save_choice == "yes":
    output_file = "grouped_data.csv"
    result.to_csv(output_file, index=False)
    print(f"✅ Aggregated data saved as {output_file}")
