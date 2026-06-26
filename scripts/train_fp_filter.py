import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

# Load task1 data
df = pd.read_csv("./training_data/task1_all.csv")

# Keep only numeric columns (no object/text types)
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove label, metadata, and derived columns from features
exclude = ['fp_label', 'campaign_id', 'stage_order', 'technique_id',
           'first_time_seen', 'target_vulnerable', 'ti_match']
feature_cols = [c for c in feature_cols if c not in exclude]

# Split by the existing split column
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

X_train, y_train = train_df[feature_cols], train_df['fp_label']
X_val, y_val = val_df[feature_cols], val_df['fp_label']
X_test, y_test = test_df[feature_cols], test_df['fp_label']

# Fill NaNs
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)

# Balance classes: weight attacks higher so the filter doesn't discard real attacks
n_benign = (y_train == 0).sum()
n_attack = (y_train == 1).sum()
ratio = max(n_benign / max(n_attack, 1), 1.0)
print(f"  Class balance: {n_benign} benign, {n_attack} attack, scale_pos_weight={ratio:.2f}")

model = xgb.XGBClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.1,
    scale_pos_weight=ratio, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Evaluate
y_pred = model.predict(X_test)
print("Test Performance:")
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Benign', 'Attack'], zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Save
joblib.dump(model, "./models/fp_filter_model.pkl")
joblib.dump(feature_cols, "./models/fp_feature_cols.pkl")
print(f"Model saved with {len(feature_cols)} numeric features") 

