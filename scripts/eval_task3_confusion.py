import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score

ENRICHED_PATH = '../archive/CIC-IDS-2017_enriched.csv'
VAL_PAIRS_PATH = '../training_data/task3_linkpred_val.csv'
TEST_PAIRS_PATH = '../training_data/task3_linkpred_test.csv'
MODEL_PATH = '../models/link_prediction_model.pkl'

print('loading enriched data...')
df_full = pd.read_csv(ENRICHED_PATH, low_memory=False)
df_full.columns = df_full.columns.str.strip()
embed_cols = [c for c in df_full.columns if c.startswith('emb_')]
cols = ['Source IP','Destination IP','Destination Port','Protocol','technique_id','first_time_seen','target_vulnerable','ti_match','campaign_id','Timestamp']
cols_to_keep = [c for c in cols if c in df_full.columns] + embed_cols
df = df_full[cols_to_keep].copy()
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')


def build_pair_features(df, idx_a, idx_b):
    a = df.loc[idx_a]
    b = df.loc[idx_b]
    f = {}
    if 'Source IP' in df.columns: f['same_src_ip'] = int(a['Source IP'] == b['Source IP'])
    if 'Destination IP' in df.columns: f['same_dst_ip'] = int(a['Destination IP'] == b['Destination IP'])
    if 'Destination Port' in df.columns: f['same_dst_port'] = int(a['Destination Port'] == b['Destination Port'])
    if 'Protocol' in df.columns: f['same_protocol'] = int(a['Protocol'] == b['Protocol'])
    if 'technique_id' in df.columns: f['same_technique'] = int(a['technique_id'] == b['technique_id'])
    if 'Timestamp' in df.columns:
        td = abs((a['Timestamp'] - b['Timestamp']).total_seconds())
        f['time_diff_seconds'] = td
        f['time_diff_log'] = np.log1p(td)
        f['a_earlier'] = int(a['Timestamp'] <= b['Timestamp'])
    if 'first_time_seen' in df.columns:
        f['a_first_time_seen'] = a['first_time_seen']
        f['b_first_time_seen'] = b['first_time_seen']
    if 'target_vulnerable' in df.columns:
        f['a_target_vulnerable'] = a['target_vulnerable']
        f['b_target_vulnerable'] = b['target_vulnerable']
    if 'ti_match' in df.columns:
        f['a_ti_match'] = a['ti_match']
        f['b_ti_match'] = b['ti_match']
    if 'first_time_seen' in df.columns:
        f['both_first_seen'] = int(a['first_time_seen'] and b['first_time_seen'])
    if 'ti_match' in df.columns:
        f['both_ti_match'] = int(a['ti_match'] and b['ti_match'])
    if embed_cols:
        emb_a = a[embed_cols].values.astype(float)
        emb_b = b[embed_cols].values.astype(float)
        dot = np.dot(emb_a, emb_b)
        na = np.linalg.norm(emb_a)
        nb = np.linalg.norm(emb_b)
        f['cosine_similarity'] = dot/(na*nb) if na>0 and nb>0 else 0.0
        f['euclidean_distance'] = np.linalg.norm(emb_a-emb_b)
    else:
        f['cosine_similarity'] = 0.0
        f['euclidean_distance'] = 0.0
    return f


def load_pairs(path):
    pairs = pd.read_csv(path)
    X, y = [], []
    for _, row in pairs.iterrows():
        ia, ib = int(row['alert_a']), int(row['alert_b'])
        if ia not in df.index or ib not in df.index:
            continue
        X.append(build_pair_features(df, ia, ib))
        y.append(int(row['label']))
    return pd.DataFrame(X), np.array(y)

X_val, y_val = load_pairs(VAL_PAIRS_PATH)
X_test, y_test = load_pairs(TEST_PAIRS_PATH)
print('val', len(X_val), 'test', len(X_test))
model = joblib.load(MODEL_PATH)
for name, X, y in [('Validation', X_val, y_val), ('Test', X_test, y_test)]:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    cm = confusion_matrix(y, y_pred)
    print('\n===', name, '===')
    print(cm)
    print('TN', cm[0,0], 'FP', cm[0,1], 'FN', cm[1,0], 'TP', cm[1,1])
    print('ROC AUC:', roc_auc_score(y, y_prob))
