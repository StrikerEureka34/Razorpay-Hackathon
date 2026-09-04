# Razorpay Hackathon — Track 04: AI Finance Controller

> Build an agent that closes one finance-ops loop across a 50+ record batch of synthetic data, reporting its match rate and the exceptions it could not resolve.

## Overview

This project implements a **multi-source reconciliation** system with:

1. **Synthetic Dataset Generator** — deterministic, seeded, zero-LLM data pipeline
2. **Scoring Harness** — precision/recall/F1 per scenario type
3. **Reconciliation Agent** — *(coming soon)*

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate the dataset (deterministic — same output every run)
python generate_dataset.py --seed 42 --num-invoices 200 --output-dir data

# Generate a held-out test set (different seed)
python generate_dataset.py --seed 99 --num-invoices 100 --output-dir data_holdout

# Score your agent's output
python score.py --predictions results/ --ground-truth data/ground_truth.csv
```

## Dataset Design

### Core Principle: *Generate the truth first, then destroy it*

The generator creates a **clean, perfectly-matched world** (every invoice has exactly one payment), saves the linkage as `ground_truth.csv`, then applies controlled corruption to the observable files while leaving the truth table untouched.

### Output Files

| File | Records | Description |
|------|---------|-------------|
| `invoices.csv` | 200 | Accounts receivable ledger (what you're owed) |
| `bank_statement.csv` | 210 | Bank transaction history (what the bank saw) |
| `processor_payouts.csv` | 28 | Razorpay/Stripe settlement records |
| `ground_truth.csv` | 198 | Reconciliation answer key with match types |

### Corruption Catalog

| Scenario | Share | What It Breaks |
|----------|-------|---------------|
| Clean 1:1 match | 35% | Nothing — baseline |
| Processor fee (2.9% + flat) | 12% | Amount equality |
| Timing lag T+2 | 10% | Date joins |
| Bundled payout (3-6 → 1) | 10% | One-to-one assumption |
| Split payment (1 → 2) | 8% | Amount equality both ways |
| Messy/missing narration | 8% | Exact string matching |
| Customer name variants | 5% | Naive entity matching |
| FX conversion with spread | 4% | Everything |
| Duplicate bank line | 3% | Precision — punishes double-matching |
| Decoy (same amount, same week) | 3% | Amount-only matchers |
| True exceptions | ~10% | Must land on exception list |

### Zero Hallucination Guarantees

- **All amounts**: `Decimal` arithmetic (no float drift)
- **All randomness**: seeded (`random.seed(42)`, `np.random.seed(42)`, `Faker.seed(42)`)
- **No LLM**: narration templates are hardcoded Python lists, filled programmatically
- **Reproducible**: re-running with same seed → byte-identical output

## Agent Output Format

Your reconciliation agent should produce:

**`results/predicted_matches.csv`**
```csv
match_id,invoice_ids,bank_txn_ids,match_type,confidence
```

**`results/predicted_exceptions.csv`**
```csv
item_type,item_id,reason
```

## Scoring

The scorer reports precision/recall/F1 broken down by scenario type:

```
Reason Code                      GT   TP   FP   FN    Prec    Rec     F1
────────────────────────────────────────────────────────────────────────
clean                            76   74    0    2   100.0%  97.4%  98.7%
processor_fee                    24   20    1    4    95.2%  83.3%  88.9%
...
────────────────────────────────────────────────────────────────────────
OVERALL MATCHING                198  170    3   28    98.3%  85.9%  91.6%
```

## Project Structure

```
├── generate_dataset.py    # Deterministic data generator (10 scenarios)
├── score.py               # Scoring harness (per-scenario metrics)
├── requirements.txt       # faker, pandas, numpy, scipy
├── data/                  # Generated dataset (seed=42)
│   ├── invoices.csv
│   ├── bank_statement.csv
│   ├── processor_payouts.csv
│   └── ground_truth.csv
└── README.md
```

## Tech Stack

- **Python 3.10+**
- `faker` — realistic Indian company names, addresses (no ML)
- `pandas` — data manipulation & CSV export
- `numpy` — seeded random distributions (log-normal amounts)
- `decimal` — exact currency arithmetic (stdlib)