# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**Primary: a finance controller working accounts receivable, every day.** They open
the Desk to work the queue of payments the system could not settle by itself:
deciding which invoice a credit belongs to, pairing the legs of a split payment,
and confirming that a genuinely unmatched item really is unmatched. This is
routine operational work, not an occasional review, so scanning speed, keyboard
flow and density outrank expression.

A secondary audience evaluates rather than operates: the Razorpay Hackathon Track
04 judges, who open it once to check that the reported match rate and exception
list are real. Their needs are served by the same evidence the controller relies
on, not by a separate persuasion surface.

## Product Purpose

Reconcile an accounts receivable ledger against a bank statement and a payment
processor settlement file, then route every item one of three ways: settled,
escalated, or asserted unmatchable. Success is a controller finishing the queue
faster than by hand, with a defensible reason recorded against every decision.

The loop is only closed when a human can act. The agent's own output is an input
to the controller's work, never the last word.

**What this build has to prove.** Two things, in order: that the solution
genuinely works on a real batch with measured accuracy and an honest account of
its failures, and that the same machinery adapts to neighbouring finance-ops
loops rather than being a one-off script. Reconciliation is the loop that is
built and measured; settlement question answering, cash forecasting and tax-line
matching are adjacent loops the same pipeline could serve. Those adjacent loops
are not built. Any surface that mentions them must say so plainly and show what
already exists that would carry them, never imply they ship today.

## Positioning

A deterministic core does all matching and all arithmetic; a language model
adjudicates only the ambiguous residue and writes the human-facing rationale. The
model never computes a fee, never enumerates candidates, and receives at most
five pre-computed options. Confidence comes from agreement between the
deterministic ranker and the model, never from the model's self-report.

Every result is quoted against a published floor: a deliberately naive amount
matcher scores 70.4% F1 on the same data, so the agent's 98.2% is reported as a
delta, not as a bare number. The exception list is an assertion the system is
willing to be measured on, not a place to put uncertainty.

## Operating Context

- Indian AR operations: NEFT, IMPS, RTGS and UPI rails, IFSC codes, INR read in
  lakh and crore, GST charged on payment gateway fees.
- Three source files arrive per period: `invoices.csv`, `bank_statement.csv`,
  `processor_payouts.csv`. A settlement credit names only a payout id, so the
  invoices behind it are reachable only by joining the payout file.
- Bank narration is truncated to 32 to 50 characters. Roughly a third of lines
  carry no usable invoice reference at all, so the payer name carries much of the
  load.
- The statement contains operating outflows (payroll, rent, GST remittance,
  vendor payments, utilities) that are not receivables. Knowing what to ignore is
  part of the job; listing one as an exception is a scored error.
- Decisions carry an owner. A settlement recorded without a named reviewer is not
  auditable.

## Capabilities and Constraints

- Item routing: settled, escalated to a model, held for a person, out of scope.
- Match types handled: one to one, one to many (split payments), many to one
  (bundled settlements), duplicate credits, FX with a dealt-rate spread,
  processor fee deductions.
- Money is `Decimal` end to end, converted once at the ingest boundary. Matching
  depends on exact equality, so float is never used, not even transiently.
- Runs offline. Model calls are cached to disk by hash of the request, and a null
  adjudicator serves as both the rules-only baseline and the network-failure
  fallback, so the fallback path is exercised on every test run.
- Scoring is exact set equality on both sides. There is no partial credit: a four
  invoice bundle with three correct scores the same as not trying.
- The dashboard is a single self-contained page generated at build time from a
  finished run. It carries no server and makes no network calls, which is why it
  survives a dead conference network.
- Undecided: whether the review station should persist decisions to a shared
  store rather than per-browser storage. Today a reviewer's decisions live only
  in their own browser and are exported as CSV.

## Brand Commitments

The product is named **Reconciliation Desk**. The name is binding and future work
must not rename it.

No other identity constraints were established. There is no requirement to carry
Razorpay branding, and no author or team credit is required on the page.

## Evidence on Hand

Real, reproducible, and measured. Nothing here is illustrative.

- `results/audit_trail.jsonl`: 309 decisions from an actual run, each keeping its
  candidates, their feature scores, and the rationale.
- `results/` also holds `predicted_matches.csv`, `predicted_exceptions.csv`,
  `review_queue.csv`, `journal_entries.csv`, `run_report.md`.
- Measured on seed 42: 98.2% matching F1, 97.7% match rate, 91.4% exception F1,
  309 decisions in roughly 4 seconds. Held out on seed 99: 95.9% F1, a 2.3 point
  gap.
- The naive floor is 70.4% F1, produced by `adversarial_audit.py`.
- The language model leg is worth +3 matches (165 to 168) with no new false
  positives, verified by ablation across both seeds.
- A controller working the queue took exception F1 from 91.4% to 100% by pairing
  two split payments the agent missed, measured by `apply_review.py`.

Absences future work must not paper over: there are no real customers, no
testimonials, no pricing, and no deployment. The data is synthetic by design,
generated deterministically so the labels are correct by construction.

## Product Principles

1. **Measured against a floor, or not claimed.** Every accuracy number is
   reported as a delta over the naive baseline on the same data.
2. **The exception list is an assertion.** Uncertainty with a plausible candidate
   goes to a person; only genuinely unmatchable items are asserted.
3. **Arithmetic is never delegated to a model.** Determinism owns money, dates
   and set membership; the model only chooses between options it was handed.
4. **Every decision keeps its evidence.** Any item can be opened to see the
   candidates considered, their scores, and who or what decided.
5. **Degrade rather than fail.** No network, no key, no problem: the rules-only
   path is a first-class configuration, not an error state.
