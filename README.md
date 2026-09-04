# Razorpay Hackathon, Track 04: AI Finance Controller

Build an agent that closes one finance-ops loop across a batch of synthetic data,
then reports its match rate and the exceptions it could not resolve.

This repo holds the benchmark that agent gets graded against: a deterministic data
generator, a scoring harness, and two audits that check the dataset is actually
hard enough to be worth scoring.

## Quick start

```bash
pip install -r requirements.txt

# Build the dataset. Same seed gives byte-identical output every time.
python generate_dataset.py --seed 42 --num-invoices 200 --output-dir data

# Check it: structural invariants, then difficulty.
python audit_data.py
python adversarial_audit.py

# Score an agent run.
python score.py --predictions results/ --ground-truth data/ground_truth.csv
```

A held-out set for your final number:

```bash
python generate_dataset.py --seed 99 --num-invoices 100 --output-dir data_holdout
```

Tune against `data/`, report against `data_holdout/`. If you tune prompts on the
same rows you report on, the number means nothing.

## How the data is made

There is no public dataset of bank lines plus invoices plus the correct pairings
between them. That pairing is exactly the label you need, and it is exactly what
nobody publishes, because it is company-specific and full of PII.

So the generator builds the truth first and then breaks it. It creates a clean
world where every invoice has one matching payment, writes that pairing table to
`ground_truth.csv`, and then runs a corruption pass over the observable files:
fees, timing, messy narrations, bundling, duplicates. The truth table is never
touched. Your agent sees only the mangled files. The scorer sees the truth.

That gives you something no real dataset can: labels that are correct by
construction, and a dial for difficulty.

## What gets generated

| File | Rows | What it is |
|------|------|------------|
| `invoices.csv` | 200 | The AR ledger, what you are owed |
| `bank_statement.csv` | 301 | 197 credits and 104 operating debits |
| `processor_payouts.csv` | 30 | Razorpay and Stripe settlements |
| `ground_truth.csv` | 305 | The answer key |

The ground truth splits into 172 matchable entries, 32 exceptions, and 101
out-of-scope lines.

`processor_payouts.csv` is a real third source, not decoration. A settlement
credit in the bank statement carries only the payout id in its narration. The
invoices behind it are reachable only by joining to the payout file and reading
`invoice_refs`. Bundled settlements cannot be solved without that hop.

## The corruption catalog

Shares are of the 204 AR-relevant ground-truth rows, so operating outflows are
excluded from the denominator.

| Scenario | Rows | Share | What it breaks |
|----------|-----:|------:|----------------|
| Clean 1:1 match | 58 | 28.4% | Nothing, this is the baseline |
| Processor fee (Razorpay 2% + GST, Stripe 2.9% + flat) | 24 | 11.8% | Amount equality |
| Timing lag | 20 | 9.8% | Date joins |
| Split payment, one invoice paid twice | 16 | 7.8% | Amount equality in both directions |
| Messy or missing reference | 16 | 7.8% | Exact string matching |
| Customer name variants | 10 | 4.9% | Naive entity matching |
| Decoy, same amount in the same week | 8 | 3.9% | Amount-only matchers |
| FX with a dealt-rate spread | 8 | 3.9% | Everything |
| Duplicate bank line | 6 | 2.9% | Precision, it punishes double-matching |
| Bundled payout, 3 or 4 invoices to one credit | 6 | 2.9% | The one-to-one assumption |
| True exceptions | 32 | 15.7% | These must land on the exception list |

Two rows matter more than the rest. The decoys stop a trivial amount-plus-date
join from scoring 90% and making the agent look pointless. The true exceptions
give the exception list real content, because if everything is matchable then an
agent that never abstains wins and the whole premise collapses.

The 32 exceptions are 20 unpaid invoices and 12 orphan bank lines. Only half the
orphans are keyword-obvious, things like interest credits, bank charges and TDS.
The other half are payment-shaped: a credit from a payer who is not a customer,
or one quoting an invoice number that was never issued. Those are the ones that
need judgement.

## Out-of-scope lines

101 bank lines are operating outflows: payroll, rent, GST remittance, vendor
payments, utilities. They carry `match_type = out_of_scope` and are neither
matchable nor exceptions.

This is deliberate. An agent that shrugs and dumps every unmatched line onto the
exception list scores 24% exception precision instead of 100%. Knowing what to
ignore is part of the job. These lines also make `running_balance` mean
something, which you need if you want to say anything about the cash position.

## Why the data does not leak the answers

A benchmark is only worth the headroom it leaves above a naive solver, so the
generator withholds the obvious shortcuts.

**Invoice status comes from aging, never from payment.** A real AR system does
not know whether the cash arrived, that is the entire point of reconciliation.
An earlier version set status from the scenario, which meant `status != 'issued'`
recovered the exception list at 100% F1. It now sits at 16% with 8.9% precision,
which is roughly the base rate.

**Transaction ids are shuffled after generation.** Ids get handed out scenario by
scenario, so without the shuffle the ordinal alone tells you the scenario and
every exception lands in the last block.

**References vary and often go missing.** Narrations use the full form
(`INV-000068`), short forms (`INV68`, `REF0068`), a bare number, or no reference
at all. The payer name appears in roughly two thirds of them, which is about
right for a real statement.

**Signs match the narration.** A line described as a bank charge is a debit. This
sounds obvious, and it was wrong before anyone checked.

## Difficulty baseline

```bash
python adversarial_audit.py
```

This builds the laziest solver that could plausibly work and reports what it
already scores:

```
exact amount, greedy nearest date          P= 87.8% R= 58.7% F1= 70.4%
fee band + payer name in narration + 75d   P= 90.2% R= 43.0% F1= 58.3%

>>> baseline to beat: 70.4% F1, headroom = 29.6%
```

Report your agent as a delta on this, not as a bare percentage. A headline of
94% is unreadable until you know that ten lines of greedy amount-matching gets
70.4%. Recall is where the room is: bundles, splits and FX all need aggregation
and a fee model, which greedy cannot do.

The baseline itself is deterministic. It sorts its candidate invoices rather than
iterating a set, because Python randomises string hashing per process and an
unsorted version swings by more than a point between runs on identical data.

`audit_data.py` covers the structural invariants and currently passes 16 of 16.
Those checks confirm the generator did what it said. The adversarial audit is
the one that can actually fail on a bad design, so run both.

## Agent output format

`results/predicted_matches.csv`

```csv
match_id,invoice_ids,bank_txn_ids,match_type,confidence
```

`results/predicted_exceptions.csv`

```csv
item_type,item_id,reason
```

Do not list out-of-scope lines as exceptions. Payroll and rent are not
receivables, and every one you list counts as a false positive.

## Scoring

`score.py` reports precision, recall and F1 broken down by scenario, which is a
far stronger result than a single headline number. Being able to say "98% on
clean, 71% on bundled, 44% on FX" shows you know where your agent actually
stands.

```
Reason Code                      GT   TP   FP   FN    Prec    Rec     F1
------------------------------------------------------------------------
clean                            58   57    0    1   100.0%  98.3%  99.1%
processor_fee                    24   20    1    4    95.2%  83.3%  88.9%
...
------------------------------------------------------------------------
OVERALL MATCHING                172  170    3    2    98.3%  98.8%  98.6%
EXCEPTION DETECTION              32   29    1    3    96.7%  90.6%  93.5%
```

The harness is sanity-checked in both directions. Replaying `ground_truth.csv`
as a set of predictions scores 100% on matching and 100% on exceptions. The same
run with all 101 operating outflows added to the exception list drops exception
precision to 24.1%.

## Reproducibility

- Every amount uses `Decimal`, so there is no float drift and no rounding rot in
  the ground truth.
- Every random draw is seeded: `random`, `numpy`, and `Faker`.
- No LLM anywhere in the pipeline. Narration templates are hardcoded Python lists
  filled programmatically. Generating templates offline is safe; generating rows
  is not, because a model will quietly drift on a balance or a fee and the labels
  rot without anyone noticing.
- Two runs at the same seed produce byte-identical CSVs. This is checked, not
  assumed.

The generator and the seed are both committed and the CSVs are not, so rebuild
them with the command in Quick start. Same seed, same bytes.

## Layout

```
generate_dataset.py    Deterministic data generator
score.py               Scoring harness, per-scenario metrics
audit_data.py          Structural invariants, 16 checks
adversarial_audit.py   Naive-baseline and label-leak audit
requirements.txt       faker, pandas, numpy, scipy
data/                  Generated dataset at seed 42 (not committed)
```

## Stack

Python 3.10 or newer, with `faker` for company names, `pandas` for CSV handling,
`numpy` for the seeded log-normal amount distribution, and `decimal` from the
standard library for exact currency arithmetic.
