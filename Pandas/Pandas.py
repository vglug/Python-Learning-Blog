import pandas as pd

# Create a sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', None, 'David'],
    'Age': [25, None, 30, 22, None],
    'City': ['New York', 'Los Angeles', None, 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# 1. Remove rows with missing values
df_cleaned = df.dropna()

# 2. Or fill missing values
# df_cleaned = df.fillna({'Name': 'Unknown', 'Age': df['Age'].mean(), 'City': 'Unknown'})

print("\nCleaned Data:")
print(df_cleaned)
