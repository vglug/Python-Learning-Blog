import pandas as pd
import glob
import os
path = r"C:\Users\Dell\Desktop\Hacktoberfest\csv files"
csv_files = glob.glob(os.path.join(path, "*.csv"))
print("CSV files found:", csv_files)
key_column = "ID"  
merged_df = None
for file in csv_files:
    print(f"Processing: {file}")
    df = pd.read_csv(file)
    print("Columns:", df.columns.tolist())  
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on=key_column, how="outer")
if merged_df is not None:
    output_path = os.path.join(path, "merged_output.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Merging complete! Saved as: {output_path}")
else:
    print("No data merged. Check CSV folder or key column.")
