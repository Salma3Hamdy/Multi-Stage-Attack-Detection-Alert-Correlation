import pandas as pd
import numpy as np
import joblib
from collections import defaultdict
from itertools import combinations

# ============================================================================
# Paths
# ============================================================================
ENRICHED_PATH = "./archive/sample_alerts.csv"
LINK_MODEL_PATH = "./models/link_prediction_model.pkl"
STAGE_MODEL_PATH = "./models/stage_prediction_model.pkl"
PHASE_ENCODER_PATH = "./models/phase_encoder.pkl"
TECH_ENCODER_PATH = "./models/tech_encoder.pkl"

# ============================================================================
# MITRE ATT&CK context
# ============================================================================
TECHNIQUE_INFO = {
    'T1046': {'name': 'Network Service Scanning', 'tactic': 'Reconnaissance', 'severity': 'LOW'},
    'T1190': {'name': 'Exploit Public-Facing Application', 'tactic': 'Initial Access', 'severity': 'HIGH'},
    'T1059': {'name': 'Command & Scripting Interpreter', 'tactic': 'Execution', 'severity': 'HIGH'},
    'T1110': {'name': 'Brute Force', 'tactic': 'Credential Access', 'severity': 'MEDIUM'},
    'T1071': {'name': 'Application Layer Protocol', 'tactic': 'Command & Control', 'severity': 'MEDIUM'},
    'T1498': {'name': 'Network Denial of Service', 'tactic': 'Impact', 'severity': 'CRITICAL'},
    'T1055': {'name': 'Process Injection', 'tactic': 'Defense Evasion', 'severity': 'HIGH'},
    'T1003': {'name': 'OS Credential Dumping', 'tactic': 'Credential Access', 'severity': 'HIGH'},
    'T1566': {'name': 'Phishing', 'tactic': 'Initial Access', 'severity': 'MEDIUM'},
    'T1021': {'name': 'Remote Services', 'tactic': 'Lateral Movement', 'severity': 'MEDIUM'},
    'T1083': {'name': 'File & Directory Discovery', 'tactic': 'Discovery', 'severity': 'LOW'},
    'BENIGN': {'name': 'Benign Activity', 'tactic': 'None', 'severity': 'LOW'},
}

KILL_CHAIN_ORDER = {
    'Reconnaissance': 1, 'Initial Access': 2, 'Execution': 3,
    'Credential Access': 4, 'Discovery': 5, 'Defense Evasion': 6,
    'Lateral Movement': 7, 'Command & Control': 8, 'Impact': 9,
}

APT_PROFILES = {
    'APT29': {'T1046', 'T1190', 'T1059', 'T1071', 'T1110', 'T1055', 'T1003', 'T1021'},
    'APT28': {'T1046', 'T1190', 'T1071', 'T1566', 'T1059', 'T1498'},
    'Lazarus': {'T1190', 'T1059', 'T1110', 'T1498', 'T1055', 'T1003', 'T1021'},
}

# ============================================================================
# 1. Load enriched data
# ============================================================================
print("Loading enriched data...")
df = pd.read_csv(ENRICHED_PATH, low_memory=False)
df.columns = df.columns.str.strip()

# Fix technique_id NaN (caused by 'None' being read as NaN from CSV)
df['technique_id'] = df['technique_id'].fillna('BENIGN')
# Fix label dash characters
df['Label'] = df['Label'].astype(str).str.strip().str.replace('\x96', '–', regex=False)

print(f"Loaded {len(df)} alerts.")

# ============================================================================
# 2. Apply false-positive filter (Task 1)
# ============================================================================
print("Applying false-positive filter...")
fp_model = joblib.load("./models/fp_filter_model.pkl")
fp_features = joblib.load("./models/fp_feature_cols.pkl")
available_features = [c for c in fp_features if c in df.columns]
numeric_features = df[available_features].select_dtypes(include=['number']).columns.tolist()

if len(numeric_features) >= 10:
    X_fp = pd.DataFrame(0, index=df.index, columns=fp_features)
    for col in numeric_features:
        X_fp[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Use lower threshold (0.3) to favour attack recall over precision
    proba = fp_model.predict_proba(X_fp)[:, 1]
    df['fp_pred'] = (proba >= 0.3).astype(int)
    before = len(df)
    df = df[df['fp_pred'] == 1].copy()
    print(f"  FP filter (ML model): {before} alerts -> {len(df)} attack alerts "
          f"(removed {before - len(df)} likely benign)")
else:
    before = len(df)
    df = df[df['Label'] != 'BENIGN'].copy()
    print(f"  FP filter (label-based): {before} alerts -> {len(df)} attack alerts "
          f"(removed {before - len(df)} BENIGN)")

# ============================================================================
# 3. Load models and encoders
# ============================================================================
print("Loading models...")
link_model = joblib.load(LINK_MODEL_PATH)
import os
if os.path.exists("./models/stage_prediction_model.pkl"):
    stage_model = joblib.load("./models/stage_prediction_model.pkl")
else:
    stage_model = None
phase_encoder = joblib.load(PHASE_ENCODER_PATH)
tech_encoder = joblib.load(TECH_ENCODER_PATH)

# ============================================================================
# 4. Select a few SMALL test campaigns to demonstrate
# ============================================================================
campaign_counts = df['campaign_id'].value_counts()
small_campaigns = campaign_counts[(campaign_counts >= 3) & (campaign_counts <= 30)]
test_campaigns = small_campaigns.index[:3]
demo_df = df[df['campaign_id'].isin(test_campaigns)].copy()
print(f"Selected {len(test_campaigns)} small campaigns ({len(demo_df)} alerts total).")

# ============================================================================
# 5. Helper functions
# ============================================================================
def build_link_features(df, idx_a, idx_b):
    a = df.loc[idx_a]
    b = df.loc[idx_b]
    features = {}
    features['same_src_ip'] = int(a['Source IP'] == b['Source IP'])
    features['same_dst_ip'] = int(a['Destination IP'] == b['Destination IP'])
    features['same_dst_port'] = int(a['Destination Port'] == b['Destination Port'])
    features['same_protocol'] = int(a['Protocol'] == b['Protocol'])
    features['same_technique'] = int(a['technique_id'] == b['technique_id'])

    time_diff = abs((pd.to_datetime(a['Timestamp']) - pd.to_datetime(b['Timestamp'])).total_seconds())
    features['time_diff_seconds'] = time_diff
    features['time_diff_log'] = np.log1p(time_diff)
    features['a_earlier'] = int(pd.to_datetime(a['Timestamp']) <= pd.to_datetime(b['Timestamp']))

    features['a_first_time_seen'] = a['first_time_seen']
    features['b_first_time_seen'] = b['first_time_seen']
    features['a_target_vulnerable'] = a['target_vulnerable']
    features['b_target_vulnerable'] = b['target_vulnerable']
    features['a_ti_match'] = a['ti_match']
    features['b_ti_match'] = b['ti_match']
    features['both_first_seen'] = int(a['first_time_seen'] and b['first_time_seen'])
    features['both_ti_match'] = int(a['ti_match'] and b['ti_match'])
    features['cosine_similarity'] = 0.0
    features['euclidean_distance'] = 0.0
    return features


def group_alerts_into_campaigns(alerts_df, link_model, threshold=0.5):
    indices = alerts_df.index.tolist()
    graph = defaultdict(set)
    for i, j in combinations(indices, 2):
        feats = build_link_features(alerts_df, i, j)
        X = pd.DataFrame([feats])
        prob = link_model.predict_proba(X)[0, 1]
        if prob > threshold:
            graph[i].add(j)
            graph[j].add(i)

    visited = set()
    campaigns = []
    for idx in indices:
        if idx not in visited:
            stack = [idx]
            comp = []
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    comp.append(node)
                    stack.extend(graph[node] - visited)
            comp.sort(key=lambda x: pd.to_datetime(alerts_df.loc[x, 'Timestamp']))
            campaigns.append(comp)
    return campaigns


ALL_TECHNIQUE_IDS = ['T1046', 'T1190', 'T1059', 'T1110', 'T1071',
                     'T1498', 'T1055', 'T1003', 'T1566', 'T1021',
                     'T1083', 'BENIGN']

TECHNIQUE_TO_TACTIC = {
    'T1046': 'Reconnaissance', 'T1190': 'Initial Access',
    'T1059': 'Execution', 'T1110': 'Credential Access',
    'T1071': 'Command & Control', 'T1498': 'Impact',
    'T1055': 'Defense Evasion', 'T1003': 'Credential Access',
    'T1566': 'Initial Access', 'T1021': 'Lateral Movement',
    'T1083': 'Discovery', 'BENIGN': 'None',
}

def _encode_sequence_for_model(techs):
    attack_techs = [t for t in techs if t not in ('BENIGN', 'Unknown', 'None', '')]
    features = {}
    for tid in ALL_TECHNIQUE_IDS:
        features[f'count_{tid}'] = sum(1 for t in techs if t == tid)
    features['seq_length'] = len(techs)
    features['attack_count'] = len(attack_techs)
    features['benign_count'] = sum(1 for t in techs if t == 'BENIGN')
    features['attack_ratio'] = len(attack_techs) / max(len(techs), 1)
    features['unique_techniques'] = len(set(attack_techs))
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
    if len(stage_orders) >= 2:
        increasing = sum(1 for i in range(1, len(stage_orders))
                         if stage_orders[i] >= stage_orders[i-1])
        features['progression_score'] = increasing / (len(stage_orders) - 1)
    else:
        features['progression_score'] = 0.0
    if attack_techs:
        last_tactic = TECHNIQUE_TO_TACTIC.get(attack_techs[-1], 'Other')
        features['last_stage_order'] = KILL_CHAIN_ORDER.get(last_tactic, 0)
    else:
        features['last_stage_order'] = 0
    return features


def predict_stage(all_techniques):
    """Predict attack stage using XGBoost model, with rule-based fallback."""
    attack_techs = [t for t in all_techniques if t not in ('BENIGN', 'None')]
    if not attack_techs:
        return 'Unknown', 0.0

    if stage_model is not None:
        if isinstance(tech_encoder, dict):
            known = set(tech_encoder.keys())
        else:
            known = set(getattr(tech_encoder, 'classes_', []))
        real_vocab = known - {'Unknown', 'BENIGN', 'None', ''}

        if real_vocab:
            feats = _encode_sequence_for_model(all_techniques)
            X = pd.DataFrame([feats])
            probs = stage_model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            phase = phase_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            if phase != 'Benign' or not attack_techs:
                return phase, confidence

    # Rule-based fallback
    tactic_sequence = []
    for t in attack_techs:
        info = TECHNIQUE_INFO.get(t, {})
        tactic = info.get('tactic', 'Unknown')
        if tactic not in ('None', 'Unknown'):
            tactic_sequence.append(tactic)

    if not tactic_sequence:
        return 'Unknown', 0.0

    max_order = 0
    current_stage = tactic_sequence[-1]
    for tactic in tactic_sequence:
        order = KILL_CHAIN_ORDER.get(tactic, 0)
        if order > max_order:
            max_order = order
            current_stage = tactic

    unique_tactics = set(tactic_sequence)
    n_unique = len(unique_tactics)
    stage_coverage = min(n_unique / 4.0, 1.0)
    volume_factor = min(len(tactic_sequence) / 8.0, 1.0)

    ordered = sorted(unique_tactics, key=lambda t: KILL_CHAIN_ORDER.get(t, 0))
    if len(ordered) >= 2:
        progression = sum(1 for i in range(1, len(ordered))
                          if KILL_CHAIN_ORDER.get(ordered[i], 0) >
                             KILL_CHAIN_ORDER.get(ordered[i - 1], 0))
        progression_score = progression / (len(ordered) - 1)
    else:
        progression_score = 0.3

    confidence = 0.3 * stage_coverage + 0.3 * volume_factor + 0.4 * progression_score
    confidence = max(0.15, min(0.92, confidence))
    return current_stage, confidence


def compute_apt_similarity(campaign_techniques):
    """Compare technique profile to known APT groups (only attack techniques)."""
    attack_techs = [t for t in campaign_techniques if t not in ('BENIGN', 'None')]
    campaign_set = set(attack_techs)

    if not campaign_set:
        return 'None', 0.0, 'None'

    similarities = {}
    for apt, techs in APT_PROFILES.items():
        intersection = campaign_set & techs
        union = campaign_set | techs
        similarities[apt] = len(intersection) / len(union) if union else 0

    sorted_apt = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_apt, top_score = sorted_apt[0]
    if top_score > 0.5:
        conf = 'High'
    elif top_score > 0.2:
        conf = 'Medium'
    else:
        conf = 'Low'
    return top_apt, top_score, conf

# ============================================================================
# 6. Run the pipeline
# ============================================================================
print("\n=== Multi-Stage Attack Detection Pipeline ===\n")

campaigns = group_alerts_into_campaigns(demo_df, link_model)
print(f"Found {len(campaigns)} campaigns.\n")

for i, camp_indices in enumerate(campaigns, 1):
    camp_alerts = demo_df.loc[camp_indices]
    all_techniques = [str(t) for t in camp_alerts.sort_values('Timestamp')['technique_id'].tolist()]
    attack_techniques = [t for t in all_techniques if t not in ('BENIGN', 'None')]

    phase, phase_conf = predict_stage(attack_techniques)
    apt, apt_score, apt_conf = compute_apt_similarity(attack_techniques)

    # Determine severity from attack techniques
    severity_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    attack_severities = [TECHNIQUE_INFO.get(t, {}).get('severity', 'LOW') for t in attack_techniques]
    max_sev = max((severity_order.get(s, 1) for s in attack_severities), default=1)
    severity = {4: 'CRITICAL', 3: 'HIGH', 2: 'MEDIUM'}.get(max_sev, 'LOW')

    print(f"Campaign {i}:")
    print(f"  Alerts: {len(camp_alerts)}")
    print(f"  Severity: {severity}")
    print(f"  Attack technique chain: {' -> '.join(attack_techniques) if attack_techniques else 'None'}")
    print(f"  Current stage: {phase} (confidence: {phase_conf:.2f})")
    print(f"  Top APT match: {apt} (similarity: {apt_score:.2f}, confidence: {apt_conf})")

    if len(camp_indices) > 1:
        feats = build_link_features(demo_df, camp_indices[0], camp_indices[1])
        X_pair = pd.DataFrame([feats])
        prob = link_model.predict_proba(X_pair)[0, 1]
        importances = link_model.feature_importances_
        feat_names = X_pair.columns
        top_idx = np.argsort(importances)[-3:][::-1]
        reasons = [f"{feat_names[j]}={feats[feat_names[j]]}" for j in top_idx]
        print(f"  Link example (alerts 1-2): prob={prob:.2f}, top reasons: {', '.join(reasons)}")
    print()

print("Pipeline complete.")
