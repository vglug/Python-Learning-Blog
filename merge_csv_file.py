# File: merge_csv_by_key.py
# Task: Merge multiple CSV files by a common key (e.g., "id")
# Author: VGLUG - Linus Torvalds - 2025

import pandas as pd
import glob
import os

def merge_csv_files_by_key(folder_path, key_column, output_file):
    # Find all CSV files in the given folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in the folder!")
        return

    print(f"📂 Found {len(csv_files)} CSV files. Merging by key '{key_column}'...")

    # Read the first CSV as base
    merged_df = pd.read_csv(csv_files[0])
    print(f"✅ Loaded: {os.path.basename(csv_files[0])}")

    # Merge all remaining CSV files one by one
    for file in csv_files[1:]:
        df = pd.read_csv(file)
        print(f"🔄 Merging: {os.path.basename(file)}")
        merged_df = pd.merge(merged_df, df, on=key_column, how="outer")  # or "inner"

    # Save the merged file
    merged_df.to_csv(output_file, index=False)
    print(f"\n✅ Successfully merged files saved as: {output_file}")

# ------------------ RUN SECTION ------------------
if __name__ == "__main__":
    folder = input("Enter the folder path containing CSV files: ").strip()
    key = input("Enter the key column name to merge on (e.g., id): ").strip()
    output = input("Enter output CSV file name (e.g., merged_output.csv): ").strip()

    merge_csv_files_by_key(folder, key, output)
# File: merge_csv_by_key.py
# Task: Merge multiple CSV files by a common key (e.g., "id")
# Author: VGLUG - Linus Torvalds - 2025

import pandas as pd
import glob
import os

def merge_csv_files_by_key(folder_path, key_column, output_file):
    # Find all CSV files in the given folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in the folder!")
        return

    print(f"📂 Found {len(csv_files)} CSV files. Merging by key '{key_column}'...")

    # Read the first CSV as base
    merged_df = pd.read_csv(csv_files[0])
    print(f"✅ Loaded: {os.path.basename(csv_files[0])}")

    # Merge all remaining CSV files one by one
    for file in csv_files[1:]:
        df = pd.read_csv(file)
        print(f"🔄 Merging: {os.path.basename(file)}")
        merged_df = pd.merge(merged_df, df, on=key_column, how="outer")  # or "inner"

    # Save the merged file
    merged_df.to_csv(output_file, index=False)
    print(f"\n✅ Successfully merged files saved as: {output_file}")

# ------------------ RUN SECTION ------------------
if __name__ == "__main__":
    folder = input("Enter the folder path containing CSV files: ").strip()
    key = input("Enter the key column name to merge on (e.g., id): ").strip()
    output = input("Enter output CSV file name (e.g., merged_output.csv): ").strip()

    merge_csv_files_by_key(folder, key, output)
