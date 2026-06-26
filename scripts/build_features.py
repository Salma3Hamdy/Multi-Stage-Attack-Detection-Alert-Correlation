import pandas as pd
import numpy as np
import glob
import os

RAW_DIR = r"C:\Users\Salma\Downloads\GeneratedLabelledFlows\TrafficLabelling"  # <-- raw files location
OUTPUT  = "./archive/CIC-IDS-2017_enriched.csv"

# 1. Load all raw daily CSV files (skip any "cleaned"/"enriched" files)
print("Loading raw daily CSV files...")
all_csv = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
raw_files = [f for f in all_csv if "cleaned" not in os.path.basename(f).lower()
                and "enriched" not in os.path.basename(f).lower()]
if not raw_files:
    raise FileNotFoundError(f"No raw CSV files found in {RAW_DIR}")

df_list = []
for f in raw_files:
    print(f"  {os.path.basename(f)}")
    df_list.append(pd.read_csv(f, low_memory=False, encoding='ISO-8859-1'))
df = pd.concat(df_list, ignore_index=True)
print(f"Combined rows: {len(df)}")

# 2. Strip spaces from column names (e.g. ' Source IP' → 'Source IP')
df.columns = df.columns.str.strip()

# 3. Verify critical columns
required = {'Source IP', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Label'}
missing = required - set(df.columns)
if missing:
    print("Missing required columns in raw files:", missing)
    raise SystemExit(1)

# 4. Basic cleaning
df.replace([np.inf, -np.inf], np.nan, inplace=True)
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    med = df[col].median()
    df[col].fillna(0 if pd.isna(med) else med, inplace=True)

df['Label'] = df['Label'].astype(str).str.strip()
df = df[~df['Label'].isin(['', 'nan'])]

# 5. Enrichment – MITRE mapping
df['Label'] = df['Label'].str.replace('\x96', '–', regex=False)

label_map = {
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
df['technique_id'] = df['Label'].map(label_map).fillna('T1190')

df['alert_text'] = df.apply(lambda r: f"Attack {r['Label']} from {r['Source IP']} to {r['Destination IP']} on port {r['Destination Port']} via protocol {r['Protocol']}", axis=1)

# Enrichment flags
seen = set()
df['first_time_seen'] = df['Source IP'].apply(lambda ip: 0 if ip in seen else (seen.add(ip) or 1))
np.random.seed(42)
dst_ips = df['Destination IP'].unique()
vuln = {ip: np.random.choice([0,1], p=[0.7,0.3]) for ip in dst_ips}
df['target_vulnerable'] = df['Destination IP'].map(vuln)
bad = set(np.random.choice(df['Source IP'].unique(), size=max(1, int(len(df['Source IP'].unique())*0.05)), replace=False))
df['ti_match'] = df['Source IP'].apply(lambda ip: 1 if ip in bad else 0)

# 6. Campaign construction (30‑minute window)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
df.dropna(subset=['Timestamp'], inplace=True)
df.sort_values(['Source IP', 'Timestamp'], inplace=True)

camp_id = 0
df['campaign_id'] = -1
df['stage_order'] = 0
window = pd.Timedelta(minutes=30)

for ip, grp in df.groupby('Source IP'):
    grp = grp.sort_values('Timestamp')
    prev = None
    cur_camp = -1
    stage = 0
    for idx, row in grp.iterrows():
        if prev is None or (row['Timestamp'] - prev) > window:
            camp_id += 1
            cur_camp = camp_id
            stage = 1
        else:
            stage += 1
        df.at[idx, 'campaign_id'] = cur_camp
        df.at[idx, 'stage_order'] = stage
        prev = row['Timestamp']

# 7. Save and verify
df.to_csv(OUTPUT, index=False)
print(f"Enriched data saved: {len(df)} rows, {df['campaign_id'].nunique()} campaigns")

test = pd.read_csv(OUTPUT, nrows=1)
present = [c for c in test.columns if 'IP' in c or 'Protocol' in c or 'Timestamp' in c]
print("Critical columns present:", present)

