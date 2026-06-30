# Multi-Stage Attack Detection via Alert Correlation

> Correlating low-level security alerts into multi-stage attack campaigns, predicting the current attack stage, and mapping activity to the MITRE ATT&CK framework.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![ML](https://img.shields.io/badge/ML-XGBoost-orange)
![Framework](https://img.shields.io/badge/MITRE-ATT%26CK-red)


![Dashboard overview](assets/screenshots/Final%20edit%20of%20THE%20PROJECT%20.png)

*The dashboard correlating low-level alerts into a critical multi-stage campaign current stage **Impact**, with the reconstructed attack chain.*

---

## The Problem

SIEM systems generate a flood of low-level alerts, most of them noise. Real attacks rarely look like a single event, they unfold across multiple stages 
The hard question:

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

- **[1] FP filter** : a lightweight model discards noisy/benign alerts before correlation, preventing downstream stages from being overwhelmed by benign alerts.
- **[2] Enrichment** : alerts are tagged with context: *Is this the first time this source IP has been seen? Is the target host actually vulnerable to the exploit being attempted? Does the hash / domain / IP match known APT indicators?* Techniques are mapped to MITRE ATT&CK.
- **[3] Correlation**: alerts are grouped into campaigns, with **explainable links** (e.g. "linked by shared source IP" or "same process"), so the analyst can see *why* two alerts were tied together.
- **[4] Stage prediction** : The current stage of an attack campaign is inferred from the detected MITRE ATT&CK techniques and their position within the cyber kill chain.
- **[5] APT similarity** : campaign techniques are compared against known APT-group profiles, with a calibrated low/medium/high confidence rather than a raw score.

---

## Screenshots

**Attack chain & APT attribution** — the system reconstructs the kill-chain order and matches the campaign's techniques against known APT profiles, showing *which* techniques matched as evidence (here: APT28 / Fancy Bear, medium confidence).

![Attack chain and APT attribution](assets/screenshots/Final%20edit%20of%20THE%20PROJECT%202.png)

**Explainable correlation evidence** — every grouping comes with the features that justified it (shared source IP, shared technique, temporal order) and their importance, so an analyst can see *why* alerts were linked.

![Correlation evidence](assets/screenshots/Final%20edit%20of%20THE%20PROJECT%203.png)

**A second campaign at a different stage** — the same pipeline surfaces a separate campaign sitting at the *Execution* stage (T1059), with its own attribution and confidence.

![Campaign at execution stage](assets/screenshots/final%20edit%20of%20THE%20PROJECT%204.png)

---

## Tech Stack

- **Python 3.14**
- **XGBoost** for classification tasks
- **pandas / scikit-learn** for data processing and evaluation
- **Streamlit** for the interactive dashboard
- **MITRE ATT&CK** technique → tactic → kill-chain mapping
- **CIC-IDS-2017** as the source dataset

---

## Dataset & Its Limits

This project uses the **CIC-IDS-2017** intrusion-detection dataset. It's a well-known benchmark, but working with it surfaced a limitation that's worth stating, because it shapes how the results should be read:

- Roughly **75% of the traffic is benign**.
- Campaigns are built by grouping alerts on *source IP + 30-minute window*.
- Out of **53,231 reconstructed campaigns, only 19 contain actual attack traffic.**

That last point matters a lot. With so few attack campaigns, a learned multi-class **stage classifier cannot be trained to produce reliable predictions** there simply aren't enough attack examples per class (some attack stages appear only once or twice in the entire dataset). A naive model trained on this data learns to do exactly one thing: predict "benign" for everything, and report a misleadingly high accuracy by doing so.

**This was caught through a data-leakage audit (see below)**

---

## What the Results Really Mean

Because of the data limitation above, the tasks are evaluated separately and reported for what they actually do:

| Component | Approach | Honest status |
|---|---|---|
| FP filter (Task 1) | XGBoost on ~80 pure flow-level features | Trained on real flow features; synthetic/leaky features explicitly excluded |
| Alert correlation (Task 3) | Explainable link rules + link-prediction model | Links are human-interpretable by design |
| Stage prediction (Task 4) | Kill-chain rule system, with an ML model as a secondary signal | **Driven primarily by the rule-based kill-chain logic** — the learned classifier is unreliable given only 19 attack campaigns |
| APT similarity (Task 5) | Jaccard similarity vs APT technique profiles | Reports matched techniques as evidence, returns *no-confidence* instead of a misleading 0-score |

**On the Stage Prediction Model**

The stage prediction model achieved high overall accuracy (94% on the validation set and nearly 100% on the test set). However, I have to stress that these results are heavily influenced by the severe class imbalance in the dataset.

More than 93% of the samples belong to the Benign class, allowing the model to obtain high overall accuracy by correctly classifying the majority class. Class-wise evaluation provides a more realistic assessment. On the validation set, for example, the model failed to identify any Reconnaissance samples (0% recall), instead predicting all 99 instances as Benign. This is also reflected in the evaluation metrics: while the weighted F1 score reached 0.91, the macro F1 score was only 0.19, indicating poor performance on the minority attack stages.

In practice, the model reliably distinguishes Benign traffic from Impact (DDoS) events but does not generalize well to the less frequent attack stages, including Reconnaissance, Credential Access, and Command and Control, due to the limited number of training examples available for these classes.

Consequently, the dashboard does not rely on the learned classifier for attack stage prediction. Instead, attack stages are inferred using a rule-based kill chain approach that maps detected techniques to their corresponding MITRE ATT&CK tactics and reports the most advanced stage observed. This design choice prioritizes interpretability and reliable behavior over reporting performance metrics that are inflated by class imbalance.


---

## Data-Leakage Audit

When the stage prediction model reported 94% validation accuracy and nearly 100% test accuracy, the results appeared unusually strong given the nature of the dataset. Rather than accepting these metrics at face value, I conducted a systematic audit to determine whether they were influenced by data leakage or evaluation bias. The audit examined campaign overlap, duplicate alerts and alert pairs, alert-index overlap, temporal overlap, sequence overlap, synthetic feature leakage, cross-task split consistency, and class imbalance.

Key findings and fixes:

- **Temporal overlap**: train/test campaigns spanned the same dates, letting the model exploit time-correlated patterns. Fixed by moving toward a chronological split (train on the past, test on the future).
- **Sequence overlap**: identical all-benign sequences appeared across train/val/test. Fixed by deduplicating sequences and excluding all-benign campaigns from stage prediction.
- **Synthetic feature leakage**: `first_time_seen`, `target_vulnerable`, and `ti_match` were synthetically generated and wouldn't exist in real deployment. Explicitly excluded from training.
- **Cross-task split inconsistency**: different scripts assigned the same campaign to different splits. Fixed with a single shared campaign-split mapping loaded by every task.
- **Class imbalance**: the root cause of the inflated stage-prediction accuracy; documented openly.

The full audit report is in [`docs/leakage_audit.md`](docs/leakage_audit.md).

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── Clean_Dataset.py               # dataset cleaning
├── build_features_test.py         # feature-build checks
├── scripts/
│   ├── build_features.py          # feature build + ATT&CK mapping
│   ├── full_rebuild.py            # full feature rebuild
│   ├── prepare_training_data.py   # attack-first campaign selection
│   ├── generate_task4_splits.py   # stage-prediction splits (group-aware)
│   ├── train_fp_filter.py         # FP filter (leak-free features)
│   ├── train_link_prediction.py   # link-prediction model
│   ├── train_stage_prediction.py  # XGBoost stage model + rule fallback
│   ├── evaluate_link_prediction.py
│   ├── eval_task3_confusion.py
│   ├── pipeline.py                # end-to-end pipeline
│   └── dashboard.py               # interactive Streamlit dashboard
├── models/                        # trained model artifacts (.pkl / .h5)
├── training_data/                 # train/val/test splits for each task
├── assets/screenshots/            # dashboard screenshots used in this README
└── docs/
    └── leakage_audit.md           # full data-leakage audit
```

---

## Getting Started

```bash

git clone https://github.com/Salma3Hamdy/Multi-Stage-Attack-Detection-Alert-Correlation.git
cd Multi-Stage-Attack-Detection-Alert-Correlation
```
```

pip install -r requirements.txt
```

A small built-in sample (archive/sample_alerts.csv) is included so you can run the dashboard straight away without downloading the full dataset, just launch the dashboard and click "Use built-in sample."
The full CIC-IDS-2017 dataset and the large enriched/training CSVs are not included due to size. To work with the complete data, download the original dataset from the official CIC source (https://www.unb.ca/cic/datasets/ids-2017.html) and run the feature-build script to regenerate the enriched data.

### Run the pipeline

```bash
python scripts/pipeline.py
```

### Launch the dashboard

```bash
streamlit run scripts/dashboard.py
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
