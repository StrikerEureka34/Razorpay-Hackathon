"""Every tunable constant lives here. Nothing is tuned to seed 42."""
from decimal import Decimal

# ── date windows (days) ──────────────────────────────────────────
# _pay_date draws its lag from a long-tailed mixture: pay-on-receipt (-term),
# early (-14..-3), on terms (-2..+7), late (+8..+30) and badly late (+31..+75),
# weighted 7/10/50/23/10. Only half of all payments land in the tight band, so
# a narrow window silently discards a third of the dataset. The longest terms
# are NET60, and `paid = max(issue, due + lag)` floors the early side at -term,
# giving a true span of [-60, +75].
WINDOW_DAYS_DEFAULT = 75
# Split legs carry the same tail plus their own extra lag.
WINDOW_DAYS_SPLIT = 80
# Duplicate pairs are 0-1 days apart.
DUPLICATE_WINDOW_DAYS = 2

# ── amount tolerances ────────────────────────────────────────────
# An FX invoice is re-denominated: amount_gross becomes the FOREIGN amount and
# the bank receives foreign * dealt_rate, where the dealt rate is the reference
# rate times a spread in [0.985, 1.015]. So the comparison needs the rate table,
# not a percentage of the invoice. 2% gives the spread rounding headroom.
FX_TOLERANCE = Decimal("0.02")
# generate_dataset.py:80. Reference rates, before the dealt spread.
FX_RATES = {
    "USD": Decimal("83.50"),
    "EUR": Decimal("91.20"),
    "GBP": Decimal("106.30"),
}

# ── scoring weights (transparent weighted sum, never learned) ────
W_REF_EXACT = 0.50
W_REF_PREFIX = 0.25
W_NAME = 0.25
W_DATE = 0.15
W_AMOUNT = 0.10

# ── escalation thresholds, on the best-vs-second-best margin ─────
TAU_AUTO = 0.25       # above: accept deterministically, no LLM call
TAU_ESCALATE = 0.08   # between ESCALATE and AUTO: send to adjudicator
TAU_REVIEW = 0.03     # below, or adjudicator abstained: review or exception

# ── limits ───────────────────────────────────────────────────────
MAX_CANDIDATES_TO_LLM = 5
BUNDLE_MAX_K = 6

# ── models ───────────────────────────────────────────────────────
# Verified live against the API on 2026-09-05. The plan's gemini-2.5-flash
# and gemini-2.5-pro both now 404 ("no longer available to new users").
# PRO_MODEL names the *escalation* tier, not necessarily a Pro-class model:
# every Pro model returns 429 RESOURCE_EXHAUSTED on this key's quota, so the
# second tier is a stronger Flash that can actually answer. Escalating into a
# model that always errors would make the tier decorative.
FLASH_MODEL = "gemini-3.5-flash"
PRO_MODEL = "gemini-3.7-flash"
CACHE_DIR = ".llm_cache"
