import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# ============================================================================
# Paths
# ============================================================================
ENRICHED_PATH = "./archive/CIC-IDS-2017_enriched.csv"
TEST_PAIRS_PATH = "./training_data/task3_linkpred_test.csv"
MODEL_PATH = "./models/link_prediction_model.pkl"

# ============================================================================
# 1. Load enriched data – find correct column names dynamically
# ============================================================================
print("Loading enriched dataset...")
df_full = pd.read_csv(ENRICHED_PATH, low_memory=False)
print("Actual columns:", df_full.columns.tolist())

# Determine correct column names (strip spaces, just like in training)
df_full.columns = df_full.columns.str.strip()

# Identify key columns by trying both variants
def find_col(col_basename):
    # The column may be 'Source IP' or ' Source IP' originally, but after stripping it's 'Source IP'
    # We'll just use the stripped version; if it exists it's fine.
    if col_basename in df_full.columns:
        return col_basename
    # fallback: maybe the file already has spaces stripped? then it's just the name
    # Actually we already stripped above, so it will be the stripped version.
    # Let's also check the original (pre-strip) if needed, but stripping solves all.
    return None

src_ip_col = 'Source IP'
dst_ip_col = 'Destination IP'
dst_port_col = 'Destination Port'
protocol_col = 'Protocol'
timestamp_col = 'Timestamp'

# Keep only needed columns (plus any embeddings)
cols_to_keep = [src_ip_col, dst_ip_col, dst_port_col, protocol_col,
                'technique_id', 'first_time_seen', 'target_vulnerable', 'ti_match',
                'campaign_id', timestamp_col]
embed_cols = [c for c in df_full.columns if c.startswith('emb_')]
cols_to_keep += embed_cols

# Check all required columns are present
missing = [c for c in cols_to_keep if c not in df_full.columns]
if missing:
    print("ERROR: Missing columns:", missing)
    print("Available columns:", df_full.columns.tolist())
    raise SystemExit(1)

df = df_full[cols_to_keep].copy()
del df_full

df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
print(f"Loaded {len(df)} alerts.")

# ============================================================================
# 2. Pair feature builder (same as training)
# ============================================================================
def build_pair_features(df, idx_a, idx_b):
    a = df.loc[idx_a]
    b = df.loc[idx_b]
    features = {}
    features['same_src_ip'] = int(a[src_ip_col] == b[src_ip_col])
    features['same_dst_ip'] = int(a[dst_ip_col] == b[dst_ip_col])
    features['same_dst_port'] = int(a[dst_port_col] == b[dst_port_col])
    features['same_protocol'] = int(a[protocol_col] == b[protocol_col])
    features['same_technique'] = int(a['technique_id'] == b['technique_id'])

    time_diff = abs((a[timestamp_col] - b[timestamp_col]).total_seconds())
    features['time_diff_seconds'] = time_diff
    features['time_diff_log'] = np.log1p(time_diff)
    features['a_earlier'] = int(a[timestamp_col] <= b[timestamp_col])

    features['a_first_time_seen'] = a['first_time_seen']
    features['b_first_time_seen'] = b['first_time_seen']
    features['a_target_vulnerable'] = a['target_vulnerable']
    features['b_target_vulnerable'] = b['target_vulnerable']
    features['a_ti_match'] = a['ti_match']
    features['b_ti_match'] = b['ti_match']
    features['both_first_seen'] = int(a['first_time_seen'] and b['first_time_seen'])
    features['both_ti_match'] = int(a['ti_match'] and b['ti_match'])

    if embed_cols:
        emb_a = a[embed_cols].values.astype(float)
        emb_b = b[embed_cols].values.astype(float)
        dot = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        features['cosine_similarity'] = dot / (norm_a * norm_b) if norm_a>0 and norm_b>0 else 0.0
        features['euclidean_distance'] = np.linalg.norm(emb_a - emb_b)
    else:
        features['cosine_similarity'] = 0.0
        features['euclidean_distance'] = 0.0
    return features

# ============================================================================
# 3. Load test pairs and build feature matrix
# ============================================================================
print("Loading test pairs...")
pairs_df = pd.read_csv(TEST_PAIRS_PATH)
X_test = []
y_test = []
for _, row in pairs_df.iterrows():
    idx_a, idx_b = int(row['alert_a']), int(row['alert_b'])
    if idx_a not in df.index or idx_b not in df.index:
        continue
    feats = build_pair_features(df, idx_a, idx_b)
    X_test.append(feats)
    y_test.append(row['label'])
X_test = pd.DataFrame(X_test)
y_test = np.array(y_test)
print(f"Built {len(X_test)} test pairs.")

# ============================================================================
# 4. Load model and predict
# ============================================================================
model = joblib.load(MODEL_PATH)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ============================================================================
# 5. Full metrics
# ============================================================================
print("\n========== Test Classification Report ==========")
print(classification_report(y_test, y_pred, target_names=['Different Campaign', 'Same Campaign']))

print("========== Confusion Matrix ==========")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives (TN): {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP): {cm[1,1]}")

print(f"\nROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ============================================================================
# 6. Feature importances (if available)
# ============================================================================
if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    print("\n========== Top 15 Feature Importances ==========")
    print(importances.sort_values(ascending=False).head(15))
    

