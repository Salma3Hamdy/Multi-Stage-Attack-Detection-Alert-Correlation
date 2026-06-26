import pandas as pd
import numpy as np
import glob
import os

# ============================================================================
# Paths
# ============================================================================
RAW_DATA_DIR = "./archive"                # folder with original daily CSVs
OUTPUT_PATH  = "./archive/CIC-IDS-2017_enriched.csv"
FILE_PATTERN = "*.csv"

# Skip any already processed files (they don't have the original columns)
SKIP_FILES = ["CIC-IDS-2017_cleaned.csv", "CIC-IDS-2017_enriched.csv"]

# ============================================================================
# 1. Load all raw CSV files and combine
# ============================================================================
print("Loading raw CSV files...")
all_files = [f for f in sorted(glob.glob(os.path.join(RAW_DATA_DIR, FILE_PATTERN)))
             if os.path.basename(f) not in SKIP_FILES]
if not all_files:
    raise FileNotFoundError("No raw CSV files found in archive/. Place the original daily files there.")

df_list = []
for file in all_files:
    print(f"  Reading {os.path.basename(file)} ...")
    df_list.append(pd.read_csv(file, low_memory=False))
df = pd.concat(df_list, ignore_index=True)
print(f"Combined raw data: {len(df)} rows")

# ============================================================================
# 2. Clean and standardise column names
# ============================================================================
# Original columns have leading spaces (e.g., ' Source IP'). Strip them.
df.columns = df.columns.str.strip()

# Check we have the critical columns
required = {'Source IP', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Label'}
missing = required - set(df.columns)
if missing:
    print("ERROR: Missing essential columns:", missing)
    print("Available columns:", df.columns.tolist())
    raise SystemExit(1)

# ============================================================================
# 3. Basic cleaning (inf, NaN, label)
# ============================================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Fill NaNs in numeric columns with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    median = df[col].median()
    if pd.isna(median):
        median = 0
    df[col].fillna(median, inplace=True)

# Clean label
df['Label'] = df['Label'].astype(str).str.strip()
df = df[df['Label'] != '']
df = df[df['Label'] != 'nan']

# ============================================================================
# 4. Enrichment – MITRE mapping
# ============================================================================
df['Label'] = df['Label'].str.replace('\x96', '–', regex=False)

label_to_tech = {
    'BENIGN': 'BENIGN',
    'Bot': 'T1071',
    'DDoS': 'T1498',
    'DoS GoldenEye': 'T1498',
    'DoS Hulk': 'T1498',
    'DoS Slowhttptest': 'T1498',
    'DoS slowloris': 'T1498',
    'FTP-Patator': 'T1110',
    'Heartbleed': 'T1190',
    'Infiltration': 'T1059',
    'PortScan': 'T1046',
    'SSH-Patator': 'T1110',
    'Web Attack – Brute Force': 'T1110',
    'Web Attack – Sql Injection': 'T1190',
    'Web Attack – XSS': 'T1190',
    'Web Attack - Brute Force': 'T1110',
    'Web Attack - Sql Injection': 'T1190',
    'Web Attack - XSS': 'T1190',
}
df['technique_id'] = df['Label'].map(label_to_tech).fillna('T1190')

# Synthetic alert text
df['alert_text'] = df.apply(lambda r: f"Attack {r['Label']} from {r['Source IP']} to {r['Destination IP']} on port {r['Destination Port']} via protocol {r['Protocol']}", axis=1)

# Enrichment flags
seen_ips = set()
df['first_time_seen'] = df['Source IP'].apply(lambda ip: 0 if ip in seen_ips else (seen_ips.add(ip) or 1))

np.random.seed(42)
unique_dst = df['Destination IP'].unique()
vuln_map = {ip: np.random.choice([0,1], p=[0.7,0.3]) for ip in unique_dst}
df['target_vulnerable'] = df['Destination IP'].map(vuln_map)

bad_ips = set(np.random.choice(df['Source IP'].unique(), size=max(1, int(len(df['Source IP'].unique())*0.05)), replace=False))
df['ti_match'] = df['Source IP'].apply(lambda ip: 1 if ip in bad_ips else 0)

# ============================================================================
# 5. Campaign construction (30‑minute window)
# ============================================================================
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
df.dropna(subset=['Timestamp'], inplace=True)
df.sort_values(['Source IP', 'Timestamp'], inplace=True)

WINDOW = pd.Timedelta(minutes=30)
camp_id = 0
df['campaign_id'] = -1
df['stage_order'] = 0

for ip, grp in df.groupby('Source IP'):
    grp = grp.sort_values('Timestamp')
    prev_time = None
    current_camp = -1
    stage = 0
    for idx, row in grp.iterrows():
        if prev_time is None or (row['Timestamp'] - prev_time) > WINDOW:
            camp_id += 1
            current_camp = camp_id
            stage = 1
        else:
            stage += 1
        df.at[idx, 'campaign_id'] = current_camp
        df.at[idx, 'stage_order'] = stage
        prev_time = row['Timestamp']

# ============================================================================
# 6. Save – verify columns
# ============================================================================
print("Saving enriched file...")
df.to_csv(OUTPUT_PATH, index=False)
print(f"Enriched data saved: {len(df)} rows, {df['campaign_id'].nunique()} campaigns")

# Quick check
check = pd.read_csv(OUTPUT_PATH, nrows=0)
cols_needed = [c for c in check.columns if 'IP' in c or 'Protocol' in c or 'Timestamp' in c]
print("Critical columns present:", cols_needed)
if not cols_needed:
    print("WARNING: Critical columns missing! Check the raw CSV column names.")
    