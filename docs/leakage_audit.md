# Data-Leakage Audit Report

This document records the data-leakage and data-integrity audit run on the project's
training and evaluation pipeline. The goal was to verify that the reported metrics
reflected genuine generalization rather than artifacts of how the data was split or
constructed.

## Summary of Findings

| Check | Task | Status | Severity |
|---|---|---|---|
| Campaign overlap | Task 1, 3, 4 | CLEAN | – |
| Exact row/pair duplicates | Task 1, 3 | CLEAN | – |
| Alert index overlap | Task 3 | CLEAN | – |
| Temporal overlap | Task 1 | LEAKAGE | High |
| Sequence overlap | Task 4 | LEAKAGE | Medium |
| Synthetic feature leakage | Task 1 | LEAKAGE | Medium |
| Cross-task split inconsistency | Task 1 vs 3 | LEAKAGE | High |
| Label distribution imbalance | Task 4 | WARNING | Medium |

---

## Leakage 1 - Temporal Overlap (Task 1) - High

**What's happening.** All three splits cover the same time range (2017-07-04 to
2017-07-07). Almost every test row falls within the training period.

**Why it's a problem.** The split is done by `campaign_id`, not by time. In a real IDS
deployment you train on historical data and test on future data. With temporal overlap,
the model can exploit time-of-day patterns, network-load patterns, or time-correlated
attack waves it wouldn't have access to in production — so it isn't truly predicting
unseen events.

**Fix.** Use a chronological split: sort all data by timestamp, then use the earliest
~70% for training, the next ~15% for validation, and the latest ~15% for testing. This
simulates performance on genuinely future data.

---

## Leakage 2 - Sequence Overlap (Task 4, Stage Prediction) - Medium

**What's happening.** A large share of validation and test sequences are identical to
training sequences. The overlapping ones are all-benign sequences that look the same
across different campaigns.

**Why it's a problem.** The model has effectively memorized these exact inputs during
training, so it gets a "free" correct answer at evaluation time — recall, not
generalization. This inflates the reported accuracy beyond what the model would achieve
on genuinely novel sequences.

**Fix.** Deduplicate `(sequence, target)` pairs before splitting, and exclude all-benign
campaigns from stage prediction entirely, since they add no signal.

---

## Leakage 3 - Synthetic Feature Leakage (Task 1) - Medium

**What's happening.** Three features in the FP-filter training data are synthetically
generated: `first_time_seen`, `target_vulnerable`, and `ti_match`.

**Why it's a problem.** These wouldn't exist in real deployment data — they depend on
processing order or are randomly assigned rather than measured. A model that leans on
them would fail in production.

**Fix.** These columns are explicitly excluded from training, so the FP filter relies
only on genuine flow-level features.

---

## Leakage 4 - Cross-Task Split Inconsistency (Task 1 vs Task 3) - High

**What's happening.** The scripts that prepare data for different tasks used independent
campaign-split logic, so a campaign could land in the training set for one task and the
validation set for another.

**Why it's a problem.** Evaluated end-to-end, an earlier stage may have already "seen"
alerts that a later stage is being validated on, giving an unfair advantage.

**Fix.** Use a single shared campaign-split mapping, saved once and loaded by every
task's data-generation script, so the same campaign always lands in the same split.

---

## Leakage 5 - Extreme Class Imbalance (Task 4) - Warning

**What's happening.** The stage-prediction training data is overwhelmingly benign, with
only a handful of samples for some attack stages.

**Why it's a problem.** With so few examples per attack class, the model can't
meaningfully learn to distinguish stages. A high overall accuracy is driven almost
entirely by correctly predicting "benign" — the model essentially learns "always predict
benign."

**Root cause.** Only 19 of 53,231 campaigns contain attacks. After splitting, each split
gets very few attack campaigns, and most sliding windows in mixed campaigns have a benign
target.

**Fix / mitigation.** Stage prediction is driven by a rule-based kill-chain analysis
rather than the learned classifier, and the limitation is documented openly rather than
masked by a misleading accuracy figure.

---

## What's Clean

- Campaign-level splits are correct — no `campaign_id` appears in multiple splits within
  the same task.
- No exact row duplicates across Task 1 splits.
- No alert-index sharing across Task 3 splits.
- Link-prediction labels are balanced (50/50 positive/negative) across splits.
- IP overlap is moderate and expected for a network dataset where some servers are
  contacted by many hosts.
