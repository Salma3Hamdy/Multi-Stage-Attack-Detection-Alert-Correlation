import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import defaultdict
from itertools import combinations

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="Multi‑Stage Attack Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Black & Green CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    .stApp { background-color: #000000; }
    html, body, [class*="css"] { font-family: 'Share Tech Mono', monospace; color: #00ff41; }
    h1, h2, h3, h4 { color: #00ff41 !important; font-family: 'Share Tech Mono', monospace !important; }
    .stButton > button {
        background-color: #000000; color: #00ff41; border: 1px solid #00ff41;
        font-family: 'Share Tech Mono', monospace; border-radius: 0; width: 100%;
    }
    .stButton > button:hover { background-color: #00ff41; color: #000000; }
    .stSlider > div > div > div > div { background-color: #00ff41; }
    .stDataFrame { background-color: #0a0a0a; border: 1px solid #00ff41; }
    [data-testid="stExpander"] { border: 1px solid #00ff41 !important; border-radius: 0 !important; background-color: #0a0a0a !important; }
    hr { border-color: #00ff41 !important; }
    #MainMenu, footer, header {visibility: hidden;}
    .campaign-box { border: 1px solid #00ff41; padding: 20px; margin-bottom: 20px; background-color: #0a0a0a; }
    .tech-step {
        border: 1px solid #00ff41; padding: 10px 16px; text-align: center;
        font-size: 11px; color: #000000; font-weight: bold; background-color: #00ff41;
    }
    .arrow { text-align: center; color: #00ff41; font-size: 20px; }
    .label-badge { border: 1px solid #00ff41; padding: 4px 12px; font-size: 11px; color: #00ff41; display: inline-block; margin-right: 6px; }
    .metric-value { font-size: 24px; color: #00ff41; font-weight: bold; }
    .metric-label { font-size: 10px; color: #009900; letter-spacing: 1px; text-transform: uppercase; }
    .stage-indicator {
        border: 2px solid #00ff41; padding: 8px 16px; text-align: center;
        font-size: 14px; font-weight: bold; color: #00ff41; background-color: #000000;
    }
    .confidence-high { color: #00ff41; }
    .confidence-medium { color: #ffff00; }
    .confidence-low { color: #ff3300; }
    .context-box { border: 1px solid #00ff41; padding: 12px; margin-top: 8px; background-color: #000000; font-size: 11px; }
    .severity-critical { color: #ff0000; font-weight: bold; }
    .severity-high { color: #ff6600; font-weight: bold; }
    .severity-medium { color: #ffff00; }
    .severity-low { color: #00ff41; }
    .filter-box { border: 1px solid #00ff41; padding: 10px 16px; margin-bottom: 12px; background-color: #0a1a0a; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================
st.markdown("""
<div style="border:1px solid #00ff41;padding:12px 20px;margin-bottom:16px;">
    <span style="font-size:22px;">◈ MULTI‑STAGE ATTACK DETECTION</span>
    <span style="float:right;font-size:12px;">[ SYSTEM ONLINE ]</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Load models (cached) – includes FP filter
# ============================================================================
@st.cache_resource
def load_models():
    link_model = joblib.load("./models/link_prediction_model.pkl")
    phase_encoder = joblib.load("./models/phase_encoder.pkl")
    tech_encoder = joblib.load("./models/tech_encoder.pkl")
    fp_model = joblib.load("./models/fp_filter_model.pkl")
    fp_feature_cols = joblib.load("./models/fp_feature_cols.pkl")
    # Load XGBoost stage model (.pkl) — replaces old Keras .h5
    import os
    stage_model = None
    if os.path.exists("./models/stage_prediction_model.pkl"):
        stage_model = joblib.load("./models/stage_prediction_model.pkl")
    return link_model, stage_model, phase_encoder, tech_encoder, fp_model, fp_feature_cols

link_model, stage_model, phase_encoder, tech_encoder, fp_model, fp_feature_cols = load_models()

# ============================================================================
# Full MITRE ATT&CK context
# ============================================================================
TECHNIQUE_INFO = {
    'T1046': {'name': 'Network Service Scanning', 'tactic': 'Reconnaissance', 'severity': 'LOW', 'description': 'Scanning network services to identify open ports and vulnerable systems'},
    'T1190': {'name': 'Exploit Public-Facing Application', 'tactic': 'Initial Access', 'severity': 'HIGH', 'description': 'Exploiting vulnerabilities in internet-facing applications to gain entry'},
    'T1059': {'name': 'Command & Scripting Interpreter', 'tactic': 'Execution', 'severity': 'HIGH', 'description': 'Running malicious commands or scripts on compromised systems'},
    'T1110': {'name': 'Brute Force', 'tactic': 'Credential Access', 'severity': 'MEDIUM', 'description': 'Attempting to guess passwords through repeated login attempts'},
    'T1071': {'name': 'Application Layer Protocol', 'tactic': 'Command & Control', 'severity': 'MEDIUM', 'description': 'Using standard protocols to communicate with compromised systems'},
    'T1498': {'name': 'Network Denial of Service', 'tactic': 'Impact', 'severity': 'CRITICAL', 'description': 'Overwhelming network resources to disrupt services'},
    'T1055': {'name': 'Process Injection', 'tactic': 'Defense Evasion', 'severity': 'HIGH', 'description': 'Injecting malicious code into legitimate processes'},
    'T1003': {'name': 'OS Credential Dumping', 'tactic': 'Credential Access', 'severity': 'HIGH', 'description': 'Extracting password hashes from operating system memory'},
    'T1566': {'name': 'Phishing', 'tactic': 'Initial Access', 'severity': 'MEDIUM', 'description': 'Sending deceptive emails to trick users into revealing credentials'},
    'T1021': {'name': 'Remote Services', 'tactic': 'Lateral Movement', 'severity': 'MEDIUM', 'description': 'Moving through the network using remote desktop, SSH, or file shares'},
    'T1083': {'name': 'File & Directory Discovery', 'tactic': 'Discovery', 'severity': 'LOW', 'description': 'Enumerating files and directories to map the environment'},
    'BENIGN': {'name': 'Benign Activity', 'tactic': 'None', 'severity': 'LOW', 'description': 'Normal network traffic - no attack detected'},
}

APT_PROFILES = {
    'APT29 (Cozy Bear)': {
        'techniques': {'T1046', 'T1190', 'T1059', 'T1071', 'T1110', 'T1055', 'T1003', 'T1021'},
        'description': 'Russian state-sponsored group targeting government and healthcare. Known for stealthy, long-term operations.',
        'typical_targets': 'Government, healthcare, think tanks',
        'attack_style': 'Patient, methodical, multi-stage attacks with custom malware'
    },
    'APT28 (Fancy Bear)': {
        'techniques': {'T1046', 'T1190', 'T1071', 'T1566', 'T1059', 'T1498'},
        'description': 'Russian military intelligence group. Aggressive and fast-paced operations.',
        'typical_targets': 'Military, political organizations, media',
        'attack_style': 'Aggressive, fast exploitation, destructive attacks'
    },
    'Lazarus Group': {
        'techniques': {'T1190', 'T1059', 'T1110', 'T1498', 'T1055', 'T1003', 'T1021'},
        'description': 'North Korean state-sponsored group. Financially motivated and destructive.',
        'typical_targets': 'Banks, cryptocurrency exchanges, critical infrastructure',
        'attack_style': 'Destructive, financially motivated, multi-phase campaigns'
    },
    'APT41 (Double Dragon)': {
        'techniques': {'T1190', 'T1059', 'T1071', 'T1110', 'T1566', 'T1083', 'T1021'},
        'description': 'Chinese state-sponsored group conducting espionage and financially motivated attacks.',
        'typical_targets': 'Healthcare, telecom, technology',
        'attack_style': 'Dual-purpose: espionage combined with financial theft'
    },
    'FIN7': {
        'techniques': {'T1046', 'T1190', 'T1059', 'T1071', 'T1566', 'T1055', 'T1083'},
        'description': 'Financially motivated cybercrime group targeting hospitality and retail.',
        'typical_targets': 'Restaurants, hotels, retail',
        'attack_style': 'Sophisticated phishing, custom malware, Point-of-Sale attacks'
    }
}

KILL_CHAIN_ORDER = {
    'Reconnaissance': 1,
    'Initial Access': 2,
    'Execution': 3,
    'Credential Access': 4,
    'Discovery': 5,
    'Defense Evasion': 6,
    'Lateral Movement': 7,
    'Command & Control': 8,
    'Impact': 9,
}

# ============================================================================
# Enrichment function – fixed technique mapping
# ============================================================================
def enrich_alerts(df):
    """Add MITRE mapping and context flags to raw alerts."""
    df['Label'] = df['Label'].astype(str).str.strip()
    # Normalize dash variants (Windows-1252 \x96 and Unicode en-dash –)
    df['Label'] = df['Label'].str.replace('\x96', '–', regex=False)

    label_to_tech = {
        'BENIGN': 'BENIGN',
        'Bot': 'T1071', 'DDoS': 'T1498',
        'DoS GoldenEye': 'T1498', 'DoS Hulk': 'T1498',
        'DoS Slowhttptest': 'T1498', 'DoS slowloris': 'T1498',
        'FTP-Patator': 'T1110', 'Heartbleed': 'T1190',
        'Infiltration': 'T1059', 'PortScan': 'T1046',
        'SSH-Patator': 'T1110',
        'Web Attack – Brute Force': 'T1110',
        'Web Attack – Sql Injection': 'T1190',
        'Web Attack – XSS': 'T1190',
        # Plain-dash variants for user-uploaded data
        'Web Attack - Brute Force': 'T1110',
        'Web Attack - Sql Injection': 'T1190',
        'Web Attack - XSS': 'T1190',
    }
    df['technique_id'] = df['Label'].map(label_to_tech).fillna('T1190')

    df['technique_name'] = df['technique_id'].map(
        lambda t: TECHNIQUE_INFO.get(t, {}).get('name', t))
    df['tactic'] = df['technique_id'].map(
        lambda t: TECHNIQUE_INFO.get(t, {}).get('tactic', 'Unknown'))
    df['severity'] = df['technique_id'].map(
        lambda t: TECHNIQUE_INFO.get(t, {}).get('severity', 'LOW'))

    seen = set()
    df['first_time_seen'] = df['Source IP'].apply(
        lambda ip: 0 if ip in seen else (seen.add(ip) or 1))

    np.random.seed(42)
    dst_ips = df['Destination IP'].unique()
    vuln_map = {ip: np.random.choice([0, 1], p=[0.7, 0.3]) for ip in dst_ips}
    df['target_vulnerable'] = df['Destination IP'].map(vuln_map)
    bad = set(np.random.choice(
        df['Source IP'].unique(),
        size=max(1, int(len(df['Source IP'].unique()) * 0.05)),
        replace=False))
    df['ti_match'] = df['Source IP'].apply(lambda ip: 1 if ip in bad else 0)

    return df

# ============================================================================
# False-positive filter (Task 1)
# ============================================================================
def apply_fp_filter(df, fp_model, fp_feature_cols):
    """Remove predicted-benign alerts.  Returns (filtered_df, n_removed, method)."""
    available = [c for c in fp_feature_cols if c in df.columns]
    numeric_cols = df[available].select_dtypes(include=['number']).columns.tolist() if available else []

    if len(numeric_cols) >= 10:
        X = pd.DataFrame(0, index=df.index, columns=fp_feature_cols)
        for col in numeric_cols:
            X[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Lower threshold (0.3) to favour attack recall over precision
        proba = fp_model.predict_proba(X)[:, 1]
        preds = (proba >= 0.3).astype(int)
        attack_df = df[preds == 1].copy()
        return attack_df, len(df) - len(attack_df), 'model'

    # Fallback: label-based filtering
    attack_df = df[df['Label'] != 'BENIGN'].copy()
    return attack_df, len(df) - len(attack_df), 'label'

# ============================================================================
# Helper functions
# ============================================================================
def build_link_features(df, idx_a, idx_b):
    a = df.loc[idx_a]; b = df.loc[idx_b]
    return {
        'same_src_ip': int(a['Source IP'] == b['Source IP']),
        'same_dst_ip': int(a['Destination IP'] == b['Destination IP']),
        'same_dst_port': int(a['Destination Port'] == b['Destination Port']),
        'same_protocol': int(a['Protocol'] == b['Protocol']),
        'same_technique': int(a['technique_id'] == b['technique_id']),
        'time_diff_seconds': abs((pd.to_datetime(a['Timestamp']) - pd.to_datetime(b['Timestamp'])).total_seconds()),
        'time_diff_log': np.log1p(abs((pd.to_datetime(a['Timestamp']) - pd.to_datetime(b['Timestamp'])).total_seconds())),
        'a_earlier': int(pd.to_datetime(a['Timestamp']) <= pd.to_datetime(b['Timestamp'])),
        'a_first_time_seen': a['first_time_seen'], 'b_first_time_seen': b['first_time_seen'],
        'a_target_vulnerable': a['target_vulnerable'], 'b_target_vulnerable': b['target_vulnerable'],
        'a_ti_match': a['ti_match'], 'b_ti_match': b['ti_match'],
        'both_first_seen': int(a['first_time_seen'] and b['first_time_seen']),
        'both_ti_match': int(a['ti_match'] and b['ti_match']),
        'cosine_similarity': 0.0, 'euclidean_distance': 0.0
    }

def group_alerts_into_campaigns(alerts_df, link_model, threshold=0.5, max_pairs=5000):
    indices = alerts_df.index.tolist()
    graph = defaultdict(set)
    pair_count = 0
    for i, j in combinations(indices, 2):
        if pair_count >= max_pairs:
            break
        pair_count += 1
        feats = build_link_features(alerts_df, i, j)
        prob = link_model.predict_proba(pd.DataFrame([feats]))[0, 1]
        if prob > threshold:
            graph[i].add(j); graph[j].add(i)
    visited, campaigns = set(), []
    for idx in indices:
        if idx not in visited:
            stack = [idx]; comp = []
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node); comp.append(node)
                    stack.extend(graph[node] - visited)
            comp.sort(key=lambda x: pd.to_datetime(alerts_df.loc[x, 'Timestamp']))
            campaigns.append(comp)
    return campaigns

# ============================================================================
# Stage prediction – XGBoost model with rule-based fallback
# ============================================================================
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
    """Build the same feature vector the XGBoost stage model was trained on."""
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
    """Predict current attack stage using XGBoost model, falling back to rules."""
    attack_techs = [t for t in all_techniques if t not in ('BENIGN', 'None')]
    if not attack_techs:
        return 'Unknown', 0.0

    # Try XGBoost model if available and trained on real data
    if stage_model is not None:
        if isinstance(tech_encoder, dict):
            known = set(tech_encoder.keys())
        else:
            known = set(getattr(tech_encoder, 'classes_', []))
        real_vocab = known - {'Unknown', 'BENIGN', 'None', ''}

        if real_vocab:
            # Pass ALL techniques (incl. BENIGN) – model was trained on full sequences
            feats = _encode_sequence_for_model(all_techniques)
            X = pd.DataFrame([feats])
            probs = stage_model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            phase = phase_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            # If model predicts Benign but we have attack techniques, use rules
            if phase == 'Benign' and attack_techs:
                return _predict_stage_rules(attack_techs)
            return phase, confidence

    return _predict_stage_rules(attack_techs)


def _predict_stage_rules(techniques):
    """Rule-based stage prediction using MITRE ATT&CK kill-chain ordering."""
    tactic_sequence = []
    for t in techniques:
        info = TECHNIQUE_INFO.get(t, {})
        tactic = info.get('tactic', 'Unknown')
        if tactic not in ('None', 'Unknown'):
            tactic_sequence.append(tactic)

    if not tactic_sequence:
        return 'Unknown', 0.0

    # Current stage = most advanced tactic observed
    max_order = 0
    current_stage = tactic_sequence[-1]
    for tactic in tactic_sequence:
        order = KILL_CHAIN_ORDER.get(tactic, 0)
        if order > max_order:
            max_order = order
            current_stage = tactic

    unique_tactics = set(tactic_sequence)
    n_unique = len(unique_tactics)
    n_alerts = len(tactic_sequence)

    stage_coverage = min(n_unique / 4.0, 1.0)
    volume_factor = min(n_alerts / 8.0, 1.0)

    ordered = sorted(unique_tactics, key=lambda t: KILL_CHAIN_ORDER.get(t, 0))
    if len(ordered) >= 2:
        progression = sum(
            1 for i in range(1, len(ordered))
            if KILL_CHAIN_ORDER.get(ordered[i], 0) >
               KILL_CHAIN_ORDER.get(ordered[i - 1], 0))
        progression_score = progression / (len(ordered) - 1)
    else:
        progression_score = 0.3

    confidence = (0.3 * stage_coverage +
                  0.3 * volume_factor +
                  0.4 * progression_score)
    confidence = max(0.15, min(0.92, confidence))
    return current_stage, confidence

# ============================================================================
# APT attribution – only considers real attack techniques
# ============================================================================
def compute_apt_with_context(campaign_techniques):
    """Compare technique profile to real APT groups with detailed output."""
    attack_techs = [t for t in campaign_techniques if t not in ('BENIGN', 'None')]
    campaign_set = set(attack_techs)

    if not campaign_set:
        empty = {
            'name': 'N/A', 'score': 0.0, 'overlap': 0.0,
            'matched_techniques': set(),
            'description': 'No attack techniques detected to compare.',
            'targets': 'N/A', 'style': 'N/A'
        }
        return empty, 'NONE', [empty]

    results = []
    for apt_name, profile in APT_PROFILES.items():
        apt_techs = profile['techniques']
        intersection = campaign_set & apt_techs
        union = campaign_set | apt_techs
        jaccard = len(intersection) / len(union) if union else 0
        overlap_pct = len(intersection) / len(apt_techs) if apt_techs else 0

        results.append({
            'name': apt_name,
            'score': jaccard,
            'overlap': overlap_pct,
            'matched_techniques': intersection,
            'description': profile['description'],
            'targets': profile['typical_targets'],
            'style': profile['attack_style']
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    top = results[0]

    if top['score'] > 0.5:
        conf = 'HIGH'
    elif top['score'] > 0.2:
        conf = 'MEDIUM'
    else:
        conf = 'LOW'

    return top, conf, results[:3]

# ============================================================================
# Campaign severity – based on attack alerts only
# ============================================================================
def determine_campaign_severity(alerts_df):
    attack_alerts = alerts_df[alerts_df['technique_id'] != 'BENIGN']
    if attack_alerts.empty:
        return 'LOW'
    severity_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    max_sev = attack_alerts['severity'].map(severity_order).max()
    if max_sev >= 4: return 'CRITICAL'
    if max_sev >= 3: return 'HIGH'
    if max_sev >= 2: return 'MEDIUM'
    return 'LOW'

# ============================================================================
# Shared campaign display function
# ============================================================================
def display_campaigns(campaigns, data_df, link_model):
    """Render all campaigns with attack chain, stage, APT, and evidence."""
    for i, camp in enumerate(campaigns, 1):
        alerts = data_df.loc[camp].sort_values('Timestamp')
        all_techs = [str(t) for t in alerts['technique_id'].tolist()]
        attack_techs = [t for t in all_techs if t not in ('BENIGN', 'None')]

        phase, confidence = predict_stage(all_techs)

        apt_result, apt_conf, top_matches = compute_apt_with_context(attack_techs)
        severity = determine_campaign_severity(alerts)

        unique_sources = alerts['Source IP'].nunique()
        unique_dest = alerts['Destination IP'].nunique()
        attack_tactics = [t for t in alerts['tactic'].unique()
                          if t not in ('None', 'Unknown')]
        time_span = (pd.to_datetime(alerts['Timestamp'].iloc[-1]) -
                     pd.to_datetime(alerts['Timestamp'].iloc[0]))

        if confidence > 0.7:
            conf_class = 'confidence-high'
        elif confidence > 0.4:
            conf_class = 'confidence-medium'
        else:
            conf_class = 'confidence-low'

        sev_class = f"severity-{severity.lower()}"

        # ── Campaign header ──
        st.markdown(f"""
        <div class="campaign-box">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <span style="font-size:20px;">▸ CAMPAIGN {i}</span>
                    <span class="{sev_class}" style="margin-left:12px;font-size:14px;">[{severity}]</span>
                </div>
                <div style="text-align:right;">
                    <div class="metric-label">STAGE CONFIDENCE</div>
                    <div class="{conf_class}" style="font-size:28px;font-weight:bold;">{confidence:.1%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CURRENT STAGE", phase)
        c2.metric("ALERTS", len(camp))
        c3.metric("HOSTS", f"{unique_sources}S / {unique_dest}D")
        c4.metric("DURATION", str(time_span).split('.')[0])

        # ── Attack chain (only attack techniques) ──
        chain_steps = []
        for t in attack_techs:
            if not chain_steps or chain_steps[-1] != t:
                chain_steps.append(t)

        st.markdown("### ATTACK CHAIN")
        if chain_steps:
            cols = st.columns(min(len(chain_steps), 8))
            for j, t in enumerate(chain_steps[:8]):
                info = TECHNIQUE_INFO.get(t, {'name': t, 'tactic': 'Unknown'})
                with cols[j]:
                    st.markdown(f"""
                    <div class="tech-step">
                        {info['name']}<br>
                        <small style="font-size:9px;">{t} | {info['tactic']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            if len(chain_steps) > 8:
                st.caption(f"... and {len(chain_steps) - 8} more steps")
        else:
            st.markdown("""
            <div class="context-box">No attack techniques detected in this campaign.</div>
            """, unsafe_allow_html=True)

        tactic_display = ' → '.join(attack_tactics) if attack_tactics else 'None detected'
        st.markdown(f"""
        <div class="context-box">
            <strong>TACTICS:</strong> {tactic_display}
        </div>
        """, unsafe_allow_html=True)

        # ── APT attribution ──
        st.markdown("### APT ATTRIBUTION")
        if apt_conf != 'NONE':
            apt_col1, apt_col2 = st.columns([2, 1])
            with apt_col1:
                matched_names = ', '.join(
                    TECHNIQUE_INFO.get(t, {}).get('name', t)
                    for t in apt_result['matched_techniques']
                ) if apt_result['matched_techniques'] else 'None'
                st.markdown(f"""
                <div class="context-box">
                    <strong>TOP MATCH:</strong> {apt_result['name']}<br>
                    <strong>CONFIDENCE:</strong> {apt_conf} (score: {apt_result['score']:.2f})<br>
                    <strong>MATCHED TECHNIQUES:</strong> {matched_names}<br>
                    <strong>DESCRIPTION:</strong> {apt_result['description']}<br>
                    <strong>TYPICAL TARGETS:</strong> {apt_result['targets']}<br>
                    <strong>ATTACK STYLE:</strong> {apt_result['style']}
                </div>
                """, unsafe_allow_html=True)
            with apt_col2:
                st.markdown("**OTHER MATCHES:**")
                for match in top_matches[1:]:
                    matched_t = ', '.join(
                        TECHNIQUE_INFO.get(t, {}).get('name', t)
                        for t in match['matched_techniques']
                    ) if match['matched_techniques'] else 'None'
                    st.markdown(f"""
                    <div class="context-box" style="margin-bottom:4px;">
                        <strong>{match['name']}</strong><br>
                        Score: {match['score']:.2f} ({match['overlap']:.0%} overlap)<br>
                        <small>Matched: {matched_t}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="context-box">
                No attack techniques detected – APT attribution requires malicious activity.
            </div>
            """, unsafe_allow_html=True)

        # ── Correlation evidence ──
        if len(camp) > 1:
            st.markdown("### CORRELATION EVIDENCE")
            feats = build_link_features(data_df, camp[0], camp[1])
            imps = link_model.feature_importances_
            names = list(feats.keys())
            top_idx = np.argsort(imps)[-5:][::-1]

            for k in top_idx:
                name = names[k].replace('_', ' ').upper()
                val = feats[names[k]]
                imp = imps[k]
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;border:1px solid #00ff41;padding:4px 12px;margin-bottom:2px;font-size:11px;">
                    <span>{name}</span>
                    <span>VALUE: {val} | IMPORTANCE: {imp:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Raw alerts ──
        with st.expander(f"RAW ALERTS ({len(camp)} rows)"):
            display_df = alerts[['Timestamp', 'Source IP', 'Destination IP',
                                 'Label', 'technique_name', 'tactic', 'severity']].copy()
            display_df.columns = ['TIME', 'SRC IP', 'DST IP', 'LABEL',
                                  'TECHNIQUE', 'TACTIC', 'SEVERITY']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### UPLOAD ALERTS")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv",
        help="Must contain: Source IP, Destination IP, Destination Port, Protocol, Timestamp, Label"
    )

    st.markdown("---")
    st.markdown("### SETTINGS")
    threshold = st.slider("LINK THRESHOLD", 0.0, 1.0, 0.5, 0.05)
    max_alerts = st.number_input("MAX ALERTS TO PROCESS", 100, 100000, 5000, 500)

    st.markdown("---")
    st.markdown("### SAMPLE DATA")
    use_sample = st.button("USE BUILT-IN SAMPLE")

# ============================================================================
# Main – uploaded file
# ============================================================================
if uploaded_file is not None:
    st.markdown(f"""
    <div style="border:1px solid #00ff41;padding:8px 16px;margin-bottom:16px;">
        FILE LOADED: {uploaded_file.name}
    </div>
    """, unsafe_allow_html=True)

    raw_df = pd.read_csv(uploaded_file)
    raw_df.columns = raw_df.columns.str.strip()

    required = {'Source IP', 'Destination IP', 'Destination Port',
                'Protocol', 'Timestamp', 'Label'}
    missing = required - set(raw_df.columns)
    if missing:
        st.error(f"MISSING COLUMNS: {missing}")
        st.markdown("**Expected columns:** `Source IP`, `Destination IP`, "
                    "`Destination Port`, `Protocol`, `Timestamp`, `Label`")
    else:
        raw_df = raw_df.head(max_alerts)

        with st.spinner("ENRICHING ALERTS..."):
            enriched = enrich_alerts(raw_df)

        # ── Task 1: False-positive filter ──
        with st.spinner("APPLYING FALSE-POSITIVE FILTER..."):
            filtered, n_removed, fp_method = apply_fp_filter(
                enriched, fp_model, fp_feature_cols)

        if fp_method == 'model':
            st.markdown(f"""
            <div class="filter-box">
                ✓ FP FILTER (ML model): {n_removed} likely-benign alerts removed
                — {len(filtered)} attack alerts remain
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="filter-box">
                ✓ FP FILTER (label-based): {n_removed} BENIGN alerts removed
                — {len(filtered)} attack alerts remain
            </div>""", unsafe_allow_html=True)

        if filtered.empty:
            st.warning("NO ATTACK ALERTS remaining after filtering. "
                       "All traffic appears benign.")
        else:
            with st.spinner("RUNNING CORRELATION..."):
                campaigns = group_alerts_into_campaigns(
                    filtered, link_model, threshold)

            st.success(f"PROCESSED {len(filtered)} ATTACK ALERTS "
                       f"→ {len(campaigns)} CAMPAIGNS")

            if campaigns:
                display_campaigns(campaigns, filtered, link_model)
            else:
                st.warning("NO CAMPAIGNS DETECTED. Try lowering the link threshold.")

# ============================================================================
# Main – built-in sample
# ============================================================================
elif use_sample:
    with st.spinner("LOADING SAMPLE DATA..."):
        df = pd.read_csv("./archive/sample_alerts.csv", low_memory=False)
        df.columns = df.columns.str.strip()

        # Re-enrich to fix technique_id (stored as NaN for BENIGN in CSV)
        df = enrich_alerts(df)

        # Apply FP filter
        df, n_removed, fp_method = apply_fp_filter(df, fp_model, fp_feature_cols)

    if df.empty:
        st.warning("No attack alerts found in sample data.")
    else:
        # Select campaigns that actually contain attack techniques
        attack_alerts = df[df['technique_id'] != 'BENIGN']
        if 'campaign_id' in df.columns and not attack_alerts.empty:
            # Find campaigns with real attack alerts
            attack_camp_counts = attack_alerts['campaign_id'].value_counts()
            good_camps = attack_camp_counts[
                (attack_camp_counts >= 3) & (attack_camp_counts <= 200)].index[:5]
            if good_camps.empty:
                good_camps = attack_camp_counts.index[:5]
            demo = df[df['campaign_id'].isin(good_camps)].copy()
        else:
            demo = attack_alerts.head(200).copy() if not attack_alerts.empty else df.head(200).copy()

        if demo.empty:
            st.warning("No suitable campaigns found in the sample.")
        else:
            with st.spinner("RUNNING CORRELATION..."):
                campaigns = group_alerts_into_campaigns(
                    demo, link_model, threshold)

            st.success(
                f"SAMPLE: {len(demo)} ATTACK ALERTS → {len(campaigns)} CAMPAIGNS "
                f"(filtered {n_removed} benign)")

            if campaigns:
                display_campaigns(campaigns, demo, link_model)
            else:
                st.warning("NO CAMPAIGNS DETECTED. Try lowering the link threshold.")

# ============================================================================
# Main – no data loaded
# ============================================================================
else:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;color:#00ff41;">
        <pre style="font-size:14px;">
    ╔══════════════════════════════════════╗
    ║   MULTI‑STAGE ATTACK DETECTION       ║
    ║   ALERT CORRELATION ENGINE           ║
    ║                                      ║
    ║   MODELS: Link + Stage + FP + APT    ║
    ║   STATUS: READY                      ║
    ╚══════════════════════════════════════╝
        </pre>
        <p style="margin-top:24px;">◂ UPLOAD A CSV FILE OR CLICK 'USE BUILT-IN SAMPLE'</p>
        <p style="font-size:11px;color:#006600;">Required columns: Source IP, Destination IP, Destination Port, Protocol, Timestamp, Label</p>
    </div>
    """, unsafe_allow_html=True)

