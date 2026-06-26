import pandas as pd
import joblib
import os

"""
Generate Task 4 (stage prediction) data using the SHARED chronological split
from prepare_training_data.py.  This eliminates cross-task split inconsistency.
"""

# ============================================================================
# Configuration
# ============================================================================
ENRICHED_PATH = "./archive/CIC-IDS-2017_enriched.csv"
OUTPUT_DIR = "./training_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHUNK_SIZE = 50000

# ============================================================================
# 1. Load shared split config (created by prepare_training_data.py)
# ============================================================================
config_path = OUTPUT_DIR + "split_config.pkl"
camp_map_path = OUTPUT_DIR + "campaign_split_map.pkl"

if not os.path.exists(config_path) or not os.path.exists(camp_map_path):
    print("ERROR: Shared split files not found. Run prepare_training_data.py first.")
    raise SystemExit(1)

split_config = joblib.load(config_path)
camp_split_map = joblib.load(camp_map_path)

cutoff_train = split_config["cutoff_train"]
cutoff_val   = split_config["cutoff_val"]
print(f"Loaded shared split config:")
print(f"  Train: {split_config['ts_min']} to {cutoff_train}")
print(f"  Val:   {cutoff_train} to {cutoff_val}")
print(f"  Test:  {cutoff_val} to {split_config['ts_max']}")
print(f"  {len(camp_split_map)} campaigns with assigned splits")

# ============================================================================
# 2. Load data for campaigns that have a split assignment
# ============================================================================
print("\nStep 1/3: Loading data for assigned campaigns...")
assigned_camps = set(camp_split_map.keys())
filtered_chunks = []

for chunk in pd.read_csv(ENRICHED_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    mask = chunk["campaign_id"].isin(assigned_camps)
    available = [c for c in ["campaign_id", "stage_order", "technique_id", "Label", "Timestamp"]
                 if c in chunk.columns]
    filtered_chunks.append(chunk[mask][available])

df = pd.concat(filtered_chunks, ignore_index=True)
df["technique_id"] = df["technique_id"].fillna("BENIGN")
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
print(f"  Loaded {len(df)} rows, {df['campaign_id'].nunique()} campaigns")

# ============================================================================
# 3. Split campaigns using the shared mapping
# ============================================================================
print("\nStep 2/3: Splitting campaigns using shared mapping...")
campaign_sizes = df.groupby("campaign_id").size()
valid_campaigns = campaign_sizes[campaign_sizes >= 2].index
df_valid = df[df["campaign_id"].isin(valid_campaigns)].copy()

train_camp = [c for c in valid_campaigns if camp_split_map.get(c) == "train"]
val_camp   = [c for c in valid_campaigns if camp_split_map.get(c) == "val"]
test_camp  = [c for c in valid_campaigns if camp_split_map.get(c) == "test"]
print(f"  Train campaigns: {len(train_camp)}, Val: {len(val_camp)}, Test: {len(test_camp)}")

# ============================================================================
# 4. Map technique to phase and build sequences
# ============================================================================
technique_to_phase = {
    "T1046": "Reconnaissance",
    "T1190": "Initial Access",
    "T1059": "Execution",
    "T1110": "Credential Access",
    "T1071": "Command and Control",
    "T1498": "Impact",
    "BENIGN": "Benign",
    "None": "Benign",
}
df_valid["phase"] = df_valid["technique_id"].map(technique_to_phase).fillna("Other")

sort_col = "Timestamp" if "Timestamp" in df_valid.columns else "stage_order"


def build_sequences(df, campaign_ids, max_window=50, stride=5, max_samples_per_camp=100):
    """Build training sequences using sliding windows over campaigns."""
    seqs = []
    for cid in campaign_ids:
        camp = df[df["campaign_id"] == cid].sort_values(sort_col)
        if len(camp) < 2:
            continue
        all_techs = [str(t) if pd.notna(t) else "BENIGN" for t in camp["technique_id"].tolist()]
        all_phases = camp["phase"].tolist()

        if len(all_techs) <= max_window:
            seqs.append({
                "campaign_id": cid,
                "sequence": ",".join(all_techs[:-1]),
                "target_stage": all_phases[-1],
                "seq_length": len(all_techs) - 1,
            })
        else:
            count = 0
            for start in range(0, len(all_techs) - 2, stride):
                end = min(start + max_window, len(all_techs) - 1)
                seqs.append({
                    "campaign_id": cid,
                    "sequence": ",".join(all_techs[start:end]),
                    "target_stage": all_phases[end],
                    "seq_length": end - start,
                })
                count += 1
                if count >= max_samples_per_camp:
                    break
    return pd.DataFrame(seqs)


print("\nStep 3/3: Generating stage prediction sequences...")
for split_name, camp_list in [("train", train_camp), ("val", val_camp), ("test", test_camp)]:
    seq_df = build_sequences(df_valid, camp_list)
    seq_df.to_csv(OUTPUT_DIR + f"task4_stagepred_{split_name}.csv", index=False)
    print(f"  {split_name}: {len(seq_df)} sequences saved")

print("\nDone! Task 4 files use the same chronological split as all other tasks.")
