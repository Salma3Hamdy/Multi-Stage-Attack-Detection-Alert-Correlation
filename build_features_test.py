import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

CLEANED_PATH = r"C:\Users\Salma\Downloads\archive\CIC-IDS-2017_cleaned.csv"
OUTPUT_PATH  = os.path.join(os.getcwd(), "CIC-IDS-2017_enriched_test.csv")

# Load a small sample for testing
print("Loading cleaned data (sample of 50000 rows for testing)...")
df = pd.read_csv(CLEANED_PATH, low_memory=False, nrows=50000)
print(f"Loaded {len(df)} rows.")

# Map labels to MITRE techniques
label_to_technique = {
    'BENIGN':                     'None',
    'Bot':                        'T1071',   
    'DDoS':                       'T1498',   
    'DoS GoldenEye':              'T1498',
    'DoS Hulk':                   'T1498',
    'DoS Slowhttptest':           'T1498',
    'DoS slowloris':              'T1498',
    'FTP-Patator':                'T1110',
    'Heartbleed':                 'T1190',
    'Infiltration':               'T1059',
    'PortScan':                   'T1046',
    'SSH-Patator':                'T1110',
    'Web Attack – Brute Force':  'T1110',
    'Web Attack – Sql Injection':'T1190',
    'Web Attack – XSS':          'T1190',
}

def map_to_technique(label):
    return label_to_technique.get(label, 'T1190')   

df['technique_id'] = df[' Label'].apply(map_to_technique)

# Create alert text
def create_alert_text(row):
    return (f"Attack {row[' Label']} detected on port {row[' Destination Port']} "
            f"with flow duration {row[' Flow Duration']} ms")

df['alert_text'] = df.apply(create_alert_text, axis=1)

# Add random features (since IP/Timestamp columns don't exist in cleaned data)
df['first_time_seen'] = 0
np.random.seed(42)
df['target_vulnerable'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
df['ti_match'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])

# Scale flow features
flow_feature_cols = [
    ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
    'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max', ' Fwd Packet Length Min',
    ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
    'Bwd Packet Length Max', ' Bwd Packet Length Min',
    ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
    ' Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
    ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean',
    ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total',
    ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
    'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',
    ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s',
    ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
    ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
    'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
    ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
    ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
    ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
    ' Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',
    ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', ' Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', ' Subflow Bwd Packets', ' Subflow Fwd Bytes',
    ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
    ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
    ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
    ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
]

existing_flow_cols = [col for col in flow_feature_cols if col in df.columns]
print(f"Scaling {len(existing_flow_cols)} flow features...")

scaler = StandardScaler()
df[existing_flow_cols] = scaler.fit_transform(df[existing_flow_cols])

# Add campaign features (simplified since IP/Timestamp not available)
print("Building campaign features (Timestamp and Source IP not available in cleaned dataset)...")
df['campaign_id'] = np.arange(len(df)) // 1000  # Group every 1000 flows into a campaign
df['stage_order'] = np.arange(len(df)) % 1000

# Save enriched dataset
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nEnrichment complete!")
print(f"Output: {len(df)} flows, {df['campaign_id'].nunique()} campaigns")
print(f"Saved to: {OUTPUT_PATH}")

# Show sample of new features
print(f"\nSample enriched data:")
print(df[['technique_id', 'alert_text', 'first_time_seen', 'target_vulnerable', 'ti_match', 'campaign_id']].head())
