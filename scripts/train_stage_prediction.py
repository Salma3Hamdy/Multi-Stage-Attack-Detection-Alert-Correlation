import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os

# ============================================================================
# Paths
# ============================================================================
OUTPUT_DIR = "./training_data/"
MODEL_DIR = "./models/"
os.makedirs(MODEL_DIR, exist_ok=True)

train_path = OUTPUT_DIR + "task4_stagepred_train.csv"
val_path   = OUTPUT_DIR + "task4_stagepred_val.csv"
test_path  = OUTPUT_DIR + "task4_stagepred_test.csv"

# ============================================================================
# MITRE technique -> tactic mapping (kill-chain order)
# ============================================================================
TECHNIQUE_TO_TACTIC = {
    'T1046': 'Reconnaissance',
    'T1190': 'Initial Access',
    'T1059': 'Execution',
    'T1110': 'Credential Access',
    'T1071': 'Command and Control',
    'T1498': 'Impact',
    'T1055': 'Defense Evasion',
    'T1003': 'Credential Access',
    'T1566': 'Initial Access',
    'T1021': 'Lateral Movement',
    'T1083': 'Discovery',
    'BENIGN': 'Benign',
}

KILL_CHAIN_ORDER = {
    'Reconnaissance': 1, 'Initial Access': 2, 'Execution': 3,
    'Credential Access': 4, 'Discovery': 5, 'Defense Evasion': 6,
    'Lateral Movement': 7, 'Command and Control': 8, 'Impact': 9,
    'Benign': 0, 'Other': 0,
}

ALL_TECHNIQUE_IDS = ['T1046', 'T1190', 'T1059', 'T1110', 'T1071',
                     'T1498', 'T1055', 'T1003', 'T1566', 'T1021',
                     'T1083', 'BENIGN']

# ============================================================================
# 1. Feature engineering from technique sequences
# ============================================================================
def encode_sequence(seq_str):
    """Convert a comma-separated technique sequence into feature vector."""
    techs = [t.strip() for t in seq_str.split(',') if t.strip()]
    attack_techs = [t for t in techs if t not in ('BENIGN', 'Unknown', 'None', '')]

    features = {}

    # Bag-of-techniques counts
    for tid in ALL_TECHNIQUE_IDS:
        features[f'count_{tid}'] = sum(1 for t in techs if t == tid)

    # Sequence metadata
    features['seq_length'] = len(techs)
    features['attack_count'] = len(attack_techs)
    features['benign_count'] = sum(1 for t in techs if t == 'BENIGN')
    features['attack_ratio'] = len(attack_techs) / max(len(techs), 1)

    # Unique technique diversity
    unique_attack = set(attack_techs)
    features['unique_techniques'] = len(unique_attack)

    # Kill-chain stage features
    stage_orders = []
    for t in attack_techs:
        tactic = TECHNIQUE_TO_TACTIC.get(t, 'Other')
        order = KILL_CHAIN_ORDER.get(tactic, 0)
        if order > 0:
            stage_orders.append(order)

    features['max_stage'] = max(stage_orders) if stage_orders else 0
    features['min_stage'] = min(stage_orders) if stage_orders else 0
    features['unique_stages'] = len(set(stage_orders)) if stage_orders else 0
    features['stage_span'] = (max(stage_orders) - min(stage_orders)) if stage_orders else 0

    # Progression: do stages generally increase over time?
    if len(stage_orders) >= 2:
        increasing = sum(1 for i in range(1, len(stage_orders))
                         if stage_orders[i] >= stage_orders[i-1])
        features['progression_score'] = increasing / (len(stage_orders) - 1)
    else:
        features['progression_score'] = 0.0

    # Last technique encoding
    if attack_techs:
        last_tech = attack_techs[-1]
        last_tactic = TECHNIQUE_TO_TACTIC.get(last_tech, 'Other')
        features['last_stage_order'] = KILL_CHAIN_ORDER.get(last_tactic, 0)
    else:
        features['last_stage_order'] = 0

    return features

# ============================================================================
# 2. Load and process data
# ============================================================================
def load_and_encode(path):
    df = pd.read_csv(path)
    # Drop sequences that are all Unknown/empty
    df = df[~df['sequence'].str.fullmatch(r'(Unknown,?\s*)+')].copy()
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=str)
    X = pd.DataFrame([encode_sequence(s) for s in df['sequence']])
    y = df['target_stage'].reset_index(drop=True)
    return X, y

print("Loading and encoding data...")
X_train, y_train = load_and_encode(train_path)
X_val, y_val     = load_and_encode(val_path)
X_test, y_test   = load_and_encode(test_path)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

if X_train.empty:
    print("\nERROR: No valid training sequences found!")
    print("All sequences are 'Unknown' — the enriched CSV needs regeneration.")
    print("Run:  python scripts/build_features.py")
    print("Then: python scripts/prepare_training_data.py")
    print("Or:   python scripts/generate_task4_splits.py")
    raise SystemExit(1)

print(f"\nTarget stage distribution (train):")
print(y_train.value_counts())

# ============================================================================
# 3. Encode target labels – fit on training set, filter unseen from val/test
# ============================================================================
phase_encoder = LabelEncoder()
y_train_enc = phase_encoder.fit_transform(y_train)

known_phases = set(phase_encoder.classes_)
val_mask = y_val.isin(known_phases)
test_mask = y_test.isin(known_phases)

if not val_mask.all():
    unseen = set(y_val[~val_mask].unique())
    print(f"  Dropping {(~val_mask).sum()} val samples with unseen phases: {unseen}")
    X_val, y_val = X_val[val_mask], y_val[val_mask]
if not test_mask.all():
    unseen = set(y_test[~test_mask].unique())
    print(f"  Dropping {(~test_mask).sum()} test samples with unseen phases: {unseen}")
    X_test, y_test = X_test[test_mask], y_test[test_mask]

y_val_enc  = phase_encoder.transform(y_val)
y_test_enc = phase_encoder.transform(y_test)

num_classes = len(phase_encoder.classes_)
print(f"\nPhases ({num_classes}): {list(phase_encoder.classes_)}")

# ============================================================================
# 4. Train XGBoost classifier
# ============================================================================
print("\nTraining XGBoost stage predictor...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
)

model.fit(
    X_train, y_train_enc,
    eval_set=[(X_val, y_val_enc)],
    verbose=True
)

# ============================================================================
# 5. Evaluate
# ============================================================================
all_labels = list(range(num_classes))

print("\nValidation performance:")
y_val_pred = model.predict(X_val)
print(classification_report(y_val_enc, y_val_pred,
      labels=all_labels, target_names=phase_encoder.classes_, zero_division=0))

print("\nTest performance:")
y_test_pred = model.predict(X_test)
print(classification_report(y_test_enc, y_test_pred,
      labels=all_labels, target_names=phase_encoder.classes_, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_test_pred, labels=all_labels))

# ============================================================================
# 6. Save model and encoders
# ============================================================================
# Save as .pkl (XGBoost model, replaces old .h5 Keras model)
joblib.dump(model, MODEL_DIR + "stage_prediction_model.pkl")
joblib.dump(phase_encoder, MODEL_DIR + "phase_encoder.pkl")

# Save tech_to_idx mapping (used by dashboard for model-based prediction)
tech_to_idx = {t: i for i, t in enumerate(ALL_TECHNIQUE_IDS)}
joblib.dump(tech_to_idx, MODEL_DIR + "tech_to_idx.pkl")
joblib.dump(tech_to_idx, MODEL_DIR + "tech_encoder.pkl")

# Save feature column names for the dashboard
feature_names = list(X_train.columns)
joblib.dump(feature_names, MODEL_DIR + "stage_feature_cols.pkl")

# maxlen no longer needed but save a dummy for backwards compatibility
joblib.dump(50, MODEL_DIR + "maxlen.pkl")

print(f"\nModel saved to {MODEL_DIR}stage_prediction_model.pkl")
print(f"Phase encoder: {list(phase_encoder.classes_)}")
print(f"Features: {len(feature_names)}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_names)
print("\nTop 15 feature importances:")
print(importances.sort_values(ascending=False).head(15))
