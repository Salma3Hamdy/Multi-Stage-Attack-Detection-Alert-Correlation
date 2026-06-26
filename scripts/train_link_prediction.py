import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import os

# ============================================================================
# Paths
# ============================================================================
ENRICHED_PATH = "./archive/CIC-IDS-2017_enriched.csv"
TRAIN_PAIRS_PATH = "./training_data/task3_linkpred_train.csv"
VAL_PAIRS_PATH   = "./training_data/task3_linkpred_val.csv"
TEST_PAIRS_PATH  = "./training_data/task3_linkpred_test.csv"

MODEL_OUTPUT = "./models/link_prediction_model.pkl"
os.makedirs("./models", exist_ok=True)

# ============================================================================
# 1. Load enriched dataset – keep only columns needed for pair features
# ============================================================================
print("Loading enriched dataset...")
# Load the full dataset and strip whitespace from column names
df_full = pd.read_csv(ENRICHED_PATH, low_memory=False)
df_full.columns = df_full.columns.str.strip()

# Define columns we want to keep if they exist
desired_cols = [
    'Source IP', 'Destination IP', 'Destination Port', 'Protocol',
    'technique_id', 'first_time_seen', 'target_vulnerable', 'ti_match',
    'campaign_id', 'stage_order', 'Timestamp'
]

# Keep only columns that exist in the dataframe
cols_to_keep = [c for c in desired_cols if c in df_full.columns]

# Also keep any embedding columns if they exist (emb_0, emb_1, ...)
embed_cols = [c for c in df_full.columns if c.startswith('emb_')]
cols_to_keep += embed_cols

# Keep only available columns
df = df_full[cols_to_keep].copy()
del df_full   # free memory

print(f"Loaded {len(df)} alerts with {len(embed_cols)} embedding dimensions.")
print(f"Available columns: {cols_to_keep}")

# Convert timestamp to datetime if it exists
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
else:
    # Create a dummy timestamp based on index if not available
    df['Timestamp'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df.index, unit='s')

# ============================================================================
# 2. Define pair feature builder
# ============================================================================
def build_pair_features(df, idx_a, idx_b):
    """
    Given two alert indices, return a feature vector for the pair.
    """
    a = df.loc[idx_a]
    b = df.loc[idx_b]

    features = {}

    # --- Shared entity features (only if columns exist) ---
    if 'Source IP' in df.columns:
        features['same_src_ip'] = int(a['Source IP'] == b['Source IP'])
    if 'Destination IP' in df.columns:
        features['same_dst_ip'] = int(a['Destination IP'] == b['Destination IP'])
    if 'Destination Port' in df.columns:
        features['same_dst_port'] = int(a['Destination Port'] == b['Destination Port'])
    if 'Protocol' in df.columns:
        features['same_protocol'] = int(a['Protocol'] == b['Protocol'])
    if 'technique_id' in df.columns:
        features['same_technique'] = int(a['technique_id'] == b['technique_id'])

    # --- Temporal features ---
    if 'Timestamp' in df.columns:
        time_diff = abs((a['Timestamp'] - b['Timestamp']).total_seconds())
        features['time_diff_seconds'] = time_diff
        features['time_diff_log'] = np.log1p(time_diff)

        # Alert order: is alert_a earlier than alert_b?
        features['a_earlier'] = int(a['Timestamp'] <= b['Timestamp'])

    # --- Enrichment context features ---
    if 'first_time_seen' in df.columns:
        features['a_first_time_seen'] = a['first_time_seen']
        features['b_first_time_seen'] = b['first_time_seen']
    if 'target_vulnerable' in df.columns:
        features['a_target_vulnerable'] = a['target_vulnerable']
        features['b_target_vulnerable'] = b['target_vulnerable']
    if 'ti_match' in df.columns:
        features['a_ti_match'] = a['ti_match']
        features['b_ti_match'] = b['ti_match']

    # Combined enrichment signals
    if 'first_time_seen' in df.columns:
        features['both_first_seen'] = int(a['first_time_seen'] and b['first_time_seen'])
    if 'ti_match' in df.columns:
        features['both_ti_match'] = int(a['ti_match'] and b['ti_match'])

    # --- Embedding similarity (if embeddings exist) ---
    if embed_cols:
        emb_a = a[embed_cols].values.astype(float)
        emb_b = b[embed_cols].values.astype(float)
        # Cosine similarity
        dot = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a > 0 and norm_b > 0:
            features['cosine_similarity'] = dot / (norm_a * norm_b)
        else:
            features['cosine_similarity'] = 0.0
        # Euclidean distance
        features['euclidean_distance'] = np.linalg.norm(emb_a - emb_b)
    else:
        # If no embeddings, use dummy values
        features['cosine_similarity'] = 0.0
        features['euclidean_distance'] = 0.0

    # --- Stage order logic (only if from same campaign, but use dummy here) ---
    # We can include the absolute stage difference if campaign known, but
    # for training we don't use that as a feature to avoid data leakage.
    # However, we can compute it from the actual campaign (just for analysis, not model).
    return features

# ============================================================================
# 3. Load pair indices and build feature matrix + labels
# ============================================================================
def load_pairs_and_features(pairs_path, df):
    print(f"Loading pairs from {pairs_path}...")
    pairs_df = pd.read_csv(pairs_path)
    X_list = []
    y_list = []
    for _, row in pairs_df.iterrows():
        idx_a = int(row['alert_a'])
        idx_b = int(row['alert_b'])
        # Safety: ensure indices exist
        if idx_a not in df.index or idx_b not in df.index:
            continue
        feats = build_pair_features(df, idx_a, idx_b)
        X_list.append(feats)
        y_list.append(row['label'])
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    print(f"Built {len(X)} pair feature vectors.")
    return X, y

print("\nBuilding training features...")
X_train, y_train = load_pairs_and_features(TRAIN_PAIRS_PATH, df)
print("Building validation features...")
X_val, y_val = load_pairs_and_features(VAL_PAIRS_PATH, df)
print("Building test features...")
X_test, y_test = load_pairs_and_features(TEST_PAIRS_PATH, df)

# ============================================================================
# 4. Train XGBoost classifier
# ============================================================================
print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# ============================================================================
# 5. Evaluate on validation set
# ============================================================================
print("\nValidation performance:")
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_val_pred))
print("ROC AUC:", roc_auc_score(y_val, y_val_prob))

# ============================================================================
# 6. Evaluate on test set
# ============================================================================
print("\nTest performance:")
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_test_pred))
print("ROC AUC:", roc_auc_score(y_test, y_test_prob))

# ============================================================================
# 7. Save model
# ============================================================================
joblib.dump(model, MODEL_OUTPUT)
print(f"\nModel saved to {MODEL_OUTPUT}")

# ============================================================================
# 8. Feature importance (explainability)
# ============================================================================
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\nTop 15 feature importances:")
print(importances.sort_values(ascending=False).head(15))




