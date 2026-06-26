import pandas as pd
import numpy as np
from itertools import combinations
import joblib
import os

# ============================================================================
# Configuration
# ============================================================================
ENRICHED_PATH = "./archive/CIC-IDS-2017_enriched.csv"
alt_enriched_path = "../CIC-IDS-2017_enriched.csv" if os.path.isdir("./scripts") else "./CIC-IDS-2017_enriched.csv"
if not os.path.exists(ENRICHED_PATH):
    if os.path.exists("./CIC-IDS-2017_enriched.csv"):
        ENRICHED_PATH = "./CIC-IDS-2017_enriched.csv"
    elif os.path.exists(alt_enriched_path):
        ENRICHED_PATH = alt_enriched_path
    else:
        raise FileNotFoundError(f"Enriched dataset not found at {ENRICHED_PATH}")

OUTPUT_DIR = "./training_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 50000
RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test gets the remaining 0.15

# ============================================================================
# 1. First pass: find attack-aware chronological cutoffs
#    Split so 70% of ATTACKS fall in train, 15% in val, 15% in test.
#    A naive time split can leave 0 attacks in val if attacks are bursty.
# ============================================================================
print("Step 1/6: Scanning attack timestamps for chronological split...")
attack_timestamps = []
ts_min, ts_max = None, None

for chunk in pd.read_csv(ENRICHED_PATH, chunksize=CHUNK_SIZE, low_memory=False,
                          usecols=["Timestamp", "technique_id"]):
    chunk.columns = chunk.columns.str.strip()
    chunk["technique_id"] = chunk["technique_id"].fillna("BENIGN")
    ts = pd.to_datetime(chunk["Timestamp"], errors="coerce")
    chunk["_ts"] = ts

    valid = ts.dropna()
    if valid.empty:
        continue
    cmin, cmax = valid.min(), valid.max()
    if ts_min is None or cmin < ts_min:
        ts_min = cmin
    if ts_max is None or cmax > ts_max:
        ts_max = cmax

    attacks = chunk[~chunk["technique_id"].isin(["BENIGN", "None"])]
    attack_timestamps.extend(attacks["_ts"].dropna().tolist())

attack_timestamps.sort()
n_attacks = len(attack_timestamps)
cutoff_train = attack_timestamps[int(n_attacks * TRAIN_FRAC)]
cutoff_val   = attack_timestamps[int(n_attacks * (TRAIN_FRAC + VAL_FRAC))]

total_hours = (ts_max - ts_min).total_seconds() / 3600
print(f"  Time range: {ts_min} to {ts_max} ({total_hours:.1f} hours)")
print(f"  Attack-aware split: 70/15/15 of {n_attacks} attack alerts")
print(f"  Train:  {ts_min} to {cutoff_train}")
print(f"  Val:    {cutoff_train} to {cutoff_val}")
print(f"  Test:   {cutoff_val} to {ts_max}")

# Save cutoffs so other scripts can reuse them (fixes cross-task inconsistency)
split_config = {"cutoff_train": cutoff_train, "cutoff_val": cutoff_val,
                "ts_min": ts_min, "ts_max": ts_max}
joblib.dump(split_config, OUTPUT_DIR + "split_config.pkl")
print("  Saved split_config.pkl for cross-task consistency.")

# ============================================================================
# 2. Second pass: load data, assign splits by timestamp
# ============================================================================
print("Step 2/6: Loading and splitting data chronologically...")
filtered_chunks = []

for chunk in pd.read_csv(ENRICHED_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"], errors="coerce")
    chunk = chunk.dropna(subset=["Timestamp"])
    chunk["technique_id"] = chunk["technique_id"].fillna("BENIGN")
    filtered_chunks.append(chunk)

df = pd.concat(filtered_chunks, ignore_index=True)
del filtered_chunks
print(f"  Loaded {len(df)} rows.")

# Assign each alert to a split based on its timestamp
df["split"] = "train"
df.loc[df["Timestamp"] >= cutoff_train, "split"] = "val"
df.loc[df["Timestamp"] >= cutoff_val,   "split"] = "test"

# Also assign each campaign to a split based on its LAST alert timestamp
# (used by Tasks 3 & 4 which need campaign-level grouping)
camp_last_ts = df.groupby("campaign_id")["Timestamp"].max()
camp_split = pd.Series("train", index=camp_last_ts.index)
camp_split[camp_last_ts >= cutoff_train] = "val"
camp_split[camp_last_ts >= cutoff_val]   = "test"
camp_split_map = camp_split.to_dict()
joblib.dump(camp_split_map, OUTPUT_DIR + "campaign_split_map.pkl")

train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "val"]
test_df  = df[df["split"] == "test"]
print(f"  Alert-level split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

n_train_camps = df[df["split"]=="train"]["campaign_id"].nunique()
n_val_camps   = df[df["split"]=="val"]["campaign_id"].nunique()
n_test_camps  = df[df["split"]=="test"]["campaign_id"].nunique()
print(f"  Campaign-level split: train={n_train_camps}, val={n_val_camps}, test={n_test_camps}")

# Check attack distribution
for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    n_attack = (split["technique_id"] != "BENIGN").sum()
    print(f"  {name}: {n_attack} attacks ({n_attack/max(len(split),1)*100:.1f}%)")

# ============================================================================
# 3. Task 1 - False-Positive Filter
# ============================================================================
print("Step 3/6: Saving Task 1 data...")
df["fp_label"] = (df["Label"] != "BENIGN").astype(int)
task1_cols = ["fp_label", "split"] + [c for c in df.columns if c not in ["fp_label", "split"]]
df[task1_cols].to_csv(OUTPUT_DIR + "task1_all.csv", index=False)
print("  Task 1 data saved.")

# ============================================================================
# 4. Task 3 - Link Prediction pairs (split by campaign's last timestamp)
# ============================================================================
print("Step 4/6: Generating link-prediction pairs...")

# Only use campaigns with >= 2 alerts
campaign_sizes = df.groupby("campaign_id").size()
valid_camps = campaign_sizes[campaign_sizes >= 2].index
df_valid = df[df["campaign_id"].isin(valid_camps)].copy()

MAX_CAMPS_PER_SPLIT = 3000  # cap to keep pair generation fast

train_camp_all = [c for c in valid_camps if camp_split_map.get(c) == "train"]
val_camp_all   = [c for c in valid_camps if camp_split_map.get(c) == "val"]
test_camp_all  = [c for c in valid_camps if camp_split_map.get(c) == "test"]

# Prioritise attack-containing campaigns, then sample the rest
rng = np.random.RandomState(RANDOM_STATE)

def select_campaigns(camp_list, df, max_n):
    attack_camps = [c for c in camp_list
                    if (df[df["campaign_id"] == c]["technique_id"] != "BENIGN").any()]
    benign_camps = [c for c in camp_list if c not in set(attack_camps)]
    selected = list(attack_camps)
    remaining = max_n - len(selected)
    if remaining > 0 and benign_camps:
        n = min(remaining, len(benign_camps))
        selected.extend(rng.choice(benign_camps, size=n, replace=False).tolist())
    return selected[:max_n]

train_camp = select_campaigns(train_camp_all, df_valid, MAX_CAMPS_PER_SPLIT)
val_camp   = select_campaigns(val_camp_all, df_valid, MAX_CAMPS_PER_SPLIT)
test_camp  = select_campaigns(test_camp_all, df_valid, MAX_CAMPS_PER_SPLIT)
print(f"  Valid campaigns: train={len(train_camp_all)}, val={len(val_camp_all)}, test={len(test_camp_all)}")
print(f"  Selected for pairs: train={len(train_camp)}, val={len(val_camp)}, test={len(test_camp)}")


def generate_link_pairs(df, campaign_ids, max_pos_pairs=10):
    pairs = []
    rng = np.random.RandomState(RANDOM_STATE)
    for cid in campaign_ids:
        camp_df = df[df["campaign_id"] == cid]
        indices = camp_df.index.tolist()
        if len(indices) < 2:
            continue
        n_possible = len(indices) * (len(indices) - 1) // 2
        if n_possible <= max_pos_pairs:
            sampled_combos = list(combinations(indices, 2))
        else:
            sampled_combos = set()
            while len(sampled_combos) < max_pos_pairs:
                i, j = rng.choice(len(indices), size=2, replace=False)
                pair = (indices[min(i, j)], indices[max(i, j)])
                sampled_combos.add(pair)
            sampled_combos = list(sampled_combos)
        for idx_a, idx_b in sampled_combos:
            pairs.append({"alert_a": idx_a, "alert_b": idx_b, "label": 1})
    # Negative pairs (same count as positive)
    all_indices = df[df["campaign_id"].isin(campaign_ids)].index.tolist()
    camp_map = df["campaign_id"].to_dict()
    n_neg = len(pairs)
    neg_pairs = []
    while len(neg_pairs) < n_neg:
        idx_a = rng.choice(all_indices)
        idx_b = rng.choice(all_indices)
        if camp_map[idx_a] != camp_map[idx_b]:
            neg_pairs.append({"alert_a": idx_a, "alert_b": idx_b, "label": 0})
    pairs.extend(neg_pairs)
    return pd.DataFrame(pairs)


for split_name, camp_list in [("train", train_camp), ("val", val_camp), ("test", test_camp)]:
    pairs_df = generate_link_pairs(df_valid, camp_list)
    pairs_df.to_csv(OUTPUT_DIR + f"task3_linkpred_{split_name}.csv", index=False)
    print(f"  Task 3 {split_name}: {len(pairs_df)} pairs saved.")

# ============================================================================
# 5. Task 4 - Stage Prediction sequences (split by campaign's last timestamp)
# ============================================================================
print("Step 5/6: Generating stage-prediction sequences...")

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


def build_sequences(df, campaign_ids, max_window=50, stride=5, max_samples_per_camp=100):
    """Build training sequences using sliding windows over campaigns."""
    seqs = []
    for cid in campaign_ids:
        camp = df[df["campaign_id"] == cid].sort_values("Timestamp")
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


for split_name, camp_list in [("train", train_camp), ("val", val_camp), ("test", test_camp)]:
    seq_df = build_sequences(df_valid, camp_list)
    seq_df.to_csv(OUTPUT_DIR + f"task4_stagepred_{split_name}.csv", index=False)
    print(f"  Task 4 {split_name}: {len(seq_df)} sequences saved.")

# ============================================================================
# 6. Summary
# ============================================================================
print("\nStep 6/6: Summary")
print(f"  Split method: Chronological by timestamp")
print(f"  Train period: {ts_min} to {cutoff_train}")
print(f"  Val period:   {cutoff_train} to {cutoff_val}")
print(f"  Test period:  {cutoff_val} to {ts_max}")
print(f"  Shared split config saved to: {OUTPUT_DIR}split_config.pkl")
print(f"  Campaign split map saved to: {OUTPUT_DIR}campaign_split_map.pkl")
print(f"\nDone! All training data saved to: {OUTPUT_DIR}")
