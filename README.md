# Multi-Stage Attack Detection via Alert Correlation

> Correlating low-level security alerts into multi-stage attack campaigns, predicting the current attack stage, and mapping activity to the MITRE ATT&CK framework.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![ML](https://img.shields.io/badge/ML-XGBoost-orange)
![Framework](https://img.shields.io/badge/MITRE-ATT%26CK-red)
![Status](https://img.shields.io/badge/status-research%20project-yellow)

![Dashboard overview](assets/screenshots/01-dashboard-overview.png)

*The dashboard correlating low-level alerts into a critical multi-stage campaign — current stage **Impact**, with the reconstructed attack chain below.*

---

## The Problem

Security tools generate a flood of low-level alerts, most of them noise. Real attacks rarely look like a single event — they unfold across multiple stages (reconnaissance → initial access → command & control → impact). The hard question:

> **Can we automatically identify and group the low-level alerts that belong to the same multi-stage campaign, and predict what stage that attack has reached?**

This project is an attempt to answer that by correlating alerts, enriching them with context, and reasoning about them against the MITRE ATT&CK framework.

---

## How It Works

The system is a pipeline, not a single model. Raw alerts flow through five stages:

```
Raw alerts
   → [1] Lightweight false-positive filter
   → [2] Context enrichment & feature extraction (incl. ATT&CK mapping)
   → [3] Alert correlation / campaign grouping (explainable links)
   → [4] Stage prediction (current attack phase)
   → [5] APT-group similarity & calibrated confidence (low / medium / high)
```

Each stage adds something:

- **[1] FP filter** — a lightweight model discards common false-positive alerts before correlation, so the downstream stages aren't drowned in benign traffic.
- **[2] Enrichment** — alerts are tagged with context: *Is this the first time this source IP has been seen? Is the target host actually vulnerable to the exploit being attempted? Does the hash / domain / IP match known APT indicators?* Techniques are mapped to MITRE ATT&CK.
- **[3] Correlation** — alerts are grouped into campaigns, with **explainable links** (e.g. "linked by shared source IP" or "same process"), so a human can see *why* two alerts were tied together.
- **[4] Stage prediction** — the current phase of the campaign is inferred from the techniques present and their kill-chain ordering.
- **[5] APT similarity** — campaign techniques are compared against known APT-group profiles, with a calibrated low/medium/high confidence rather than a raw score.

---

## Screenshots

**Attack chain & APT attribution** — the system reconstructs the kill-chain order and matches the campaign's techniques against known APT profiles, showing *which* techniques matched as evidence (here: APT28 / Fancy Bear, medium confidence).

![Attack chain and APT attribution](assets/screenshots/02-attack-chain-apt.png)

**Explainable correlation evidence** — every grouping comes with the features that justified it (shared source IP, shared technique, temporal order) and their importance, so an analyst can see *why* alerts were linked.

![Correlation evidence](assets/screenshots/03-correlation-evidence.png)

**A second campaign at a different stage** — the same pipeline surfaces a separate campaign sitting at the *Execution* stage (T1059), with its own attribution and confidence.

![Campaign at execution stage](assets/screenshots/04-campaign-execution.png)

---

## Tech Stack

- **Python 3.14**
- **XGBoost** for classification tasks
- **pandas / scikit-learn** for data processing and evaluation
- **MITRE ATT&CK** technique → tactic → kill-chain mapping
- **CIC-IDS-2017** as the source dataset

---

## Dataset & an Honest Note on Its Limits

This project uses the **CIC-IDS-2017** intrusion-detection dataset. It's a well-known benchmark, but working with it surfaced a limitation worth stating plainly, because it shapes how the results should be read:

- Roughly **75% of the traffic is benign**.
- Campaigns are built by grouping alerts on *source IP + 30-minute window*.
- Out of **53,231 reconstructed campaigns, only 19 contain actual attack traffic.**

That last point matters a lot. With so few attack campaigns, a learned multi-class **stage classifier cannot be meaningfully trained** — there simply aren't enough attack examples per class (some attack stages appear only once or twice in the entire dataset). A naive model trained on this data learns to do exactly one thing: predict "benign" for everything, and report a misleadingly high accuracy by doing so.

**This was caught through a data-leakage audit (see below) and addressed honestly rather than hidden behind an inflated number.**

---

## What's Real vs What's Honest About the Results

Because of the data limitation above, the components are evaluated separately and reported for what they actually do:

| Component | Approach | Honest status |
|---|---|---|
| FP filter (Task 1) | XGBoost on ~80 pure flow-level features | Trained on real flow features; synthetic/leaky features explicitly excluded |
| Alert correlation (Task 3) | Explainable link rules + link-prediction model | Links are human-interpretable by design |
| Stage prediction (Task 4) | Kill-chain rule system, with an ML model as a secondary signal | **Driven primarily by the rule-based kill-chain logic** — the learned classifier is unreliable given only 19 attack campaigns |
| APT similarity (Task 5) | Jaccard similarity vs APT technique profiles | Reports matched techniques as evidence, returns *no-confidence* instead of a misleading 0-score |

> **Note:** report only numbers from the leak-free evaluation here. Fill in your real per-class metrics once the chronological / group-aware split has been run. Avoid headline "99% accuracy" claims — see the audit for why that figure was an artifact of class imbalance, not real performance.

---

## Data-Leakage Audit

A core part of this project was **auditing the evaluation itself** rather than trusting the first number that came out. The audit checked for campaign overlap, duplicate rows/pairs, alert-index overlap, temporal overlap, sequence overlap, synthetic-feature leakage, cross-task split consistency, and class balance.

Key findings and fixes:

- **Temporal overlap** — train/test campaigns spanned the same dates, letting the model exploit time-correlated patterns. Fixed by moving toward a chronological split (train on the past, test on the future).
- **Sequence overlap** — identical all-benign sequences appeared across train/val/test. Fixed by deduplicating sequences and excluding all-benign campaigns from stage prediction.
- **Synthetic feature leakage** — `first_time_seen`, `target_vulnerable`, and `ti_match` were synthetically generated and wouldn't exist in real deployment. Explicitly excluded from training.
- **Cross-task split inconsistency** — different scripts assigned the same campaign to different splits. Fixed with a single shared campaign-split mapping loaded by every task.
- **Class imbalance** — the root cause of the inflated stage-prediction accuracy; documented openly.

The full audit report is in [`docs/leakage_audit.md`](docs/leakage_audit.md).

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── build_features.py          # feature build + ATT&CK mapping
│   ├── prepare_training_data.py   # attack-first campaign selection
│   ├── generate_task4_splits.py   # stage-prediction splits (group-aware)
│   ├── train_fp_filter.py         # FP filter (leak-free features)
│   ├── train_stage_prediction.py  # XGBoost stage model + rule fallback
│   ├── pipeline.py                # end-to-end pipeline
│   └── dashboard.py               # interactive dashboard
├── models/                        # trained model artifacts (.pkl)
└── docs/
    └── leakage_audit.md
```

*(Adjust to match your actual file names.)*

---

## Getting Started

```bash
# clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# install dependencies
pip install -r requirements.txt
```

The CIC-IDS-2017 dataset is not included in this repo due to size. Download it from the [official CIC source](https://www.unb.ca/cic/datasets/ids-2017.html) and place the CSVs where the feature-build script expects them.

### Run the pipeline

```bash
python src/pipeline.py
```

### Launch the dashboard

```bash
python src/dashboard.py
```

---

## Limitations & Future Work

- The single biggest constraint is the dataset: 19 attack campaigns is too few to train a robust learned stage classifier. A richer dataset (or combining multiple IDS datasets) would let the ML stage model carry more of the load.
- Stage prediction currently leans on rule-based kill-chain logic; with more attack data this could become a properly learned, calibrated model.
- The chronological split is the more honest evaluation but is constrained by the dataset only spanning four days.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

---

## Acknowledgements

Built as a machine-learning project exploring multi-stage attack detection. Thanks to the MITRE ATT&CK framework and the Canadian Institute for Cybersecurity (CIC-IDS-2017).
