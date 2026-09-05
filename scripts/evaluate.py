"""Run every configuration against both datasets and write the ablation table."""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIGS = [
    ("rules only", ["--no-llm"]),
    ("rules + Gemini (cached)", []),
]
DATASETS = [
    ("seed 42 (dev)", "data", "data/ground_truth.csv"),
    ("seed 99 (held out)", "data_holdout", "data_holdout/ground_truth.csv"),
]

# score.py prints box-drawing characters; Windows consoles default to cp1252.
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(data_dir, out_dir, flags):
    subprocess.run(
        [sys.executable, "reconcile.py", "--data", data_dir, "--out", out_dir, *flags],
        check=True, capture_output=True, env=_ENV,
    )


def score(out_dir, ground_truth):
    result = subprocess.run(
        [sys.executable, "score.py", "--predictions", out_dir,
         "--ground-truth", ground_truth],
        check=True, capture_output=True, text=True, encoding="utf-8", env=_ENV,
    )

    def grab(label):
        m = re.search(rf"{label}:\s+([\d.]+)%", result.stdout)
        return m.group(1) + "%" if m else "n/a"

    def grab_exception_f1():
        m = re.search(
            r"EXCEPTION DETECTION\s+\d+\s+\d+\s+\d+\s+\d+\s+"
            r"[\d.]+%\s+[\d.]+%\s+([\d.]+)%",
            result.stdout,
        )
        return m.group(1) + "%" if m else "n/a"

    return (
        grab("MATCH RATE"), grab("PRECISION"), grab("RECALL"),
        grab("F1 SCORE"), grab_exception_f1(),
    )


rows = []
for config_name, flags in CONFIGS:
    for data_name, data_dir, ground_truth in DATASETS:
        if not os.path.exists(ground_truth):
            continue
        out_dir = f"results/_ablation_{abs(hash((config_name, data_name)))}"
        run(data_dir, out_dir, flags)
        rows.append((config_name, data_name, *score(out_dir, ground_truth)))

os.makedirs("results", exist_ok=True)
with open("results/ablation.md", "w", encoding="utf-8") as fh:
    fh.write("# Ablation\n\n")
    fh.write("| Configuration | Dataset | Match rate | Precision | Recall "
             "| Matching F1 | Exception F1 |\n")
    fh.write("|---|---|---|---|---|---|---|\n")
    for row in rows:
        fh.write("| " + " | ".join(row) + " |\n")

    # The held-out gap is the headline generalization number.
    by_key = {(c, d): f1 for c, d, _mr, _p, _r, f1, _ef1 in rows}
    fh.write("\n## Held-out gap\n\n")
    for config_name, _flags in CONFIGS:
        dev = by_key.get((config_name, "seed 42 (dev)"))
        held = by_key.get((config_name, "seed 99 (held out)"))
        if not dev or not held or "n/a" in (dev, held):
            continue
        gap = float(dev.rstrip("%")) - float(held.rstrip("%"))
        verdict = "within the 3-point budget" if abs(gap) <= 3 else "OVER BUDGET"
        fh.write(f"- **{config_name}**: {dev} -> {held}, "
                 f"gap {gap:+.1f} points ({verdict})\n")

print(open("results/ablation.md", encoding="utf-8").read())
