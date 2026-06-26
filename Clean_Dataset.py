import pandas as pd
import numpy as np
import glob
import os


DATA_DIR = r"C:\Users\Salma\Downloads\archive"               
FILE_PATTERN = "*.csv"                


def load_all_files(data_dir, file_pattern):
    all_files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} with pattern {file_pattern}")
    
    df_list = []
    for file in all_files:
        print(f"Loading {file} ...")
        df = pd.read_csv(file, low_memory=False)
        df_list.append(df)
    
    combined = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(all_files)} files, total rows: {len(combined)}")
    return combined

df = load_all_files(DATA_DIR, FILE_PATTERN)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


df.replace([np.inf, -np.inf], np.nan, inplace=True)

missing_counts = df.isnull().sum()
print("\nMissing values per column (top 10):")
print(missing_counts[missing_counts > 0].head(10))


numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = df[col].median()
    if pd.isna(median_val):
        median_val = 0
    df[col].fillna(median_val, inplace=True)

print("After filling, remaining NaNs:", df.isnull().sum().sum())


df[' Label'] = df[' Label'].astype(str).str.strip()


initial_rows = len(df)
df = df[df[' Label'] != '']
df = df[df[' Label'] != 'nan']
print(f"Dropped {initial_rows - len(df)} rows with missing/empty label")

print("\nLabel distribution:")
print(df[' Label'].value_counts())

CLEANED_PATH = os.path.join(DATA_DIR, "CIC-IDS-2017_cleaned.csv")
df.to_csv(CLEANED_PATH, index=False)
print(f"\nCleaned dataset saved to: {CLEANED_PATH}")