"""Solve Subchains Table V cells (Zhang & Preneel 2019, I(alpha) = 1 - revenue).

  python scripts/check_subchains.py OUT_JSONL RATIO GAMMA ALPHA [ALPHA ...]

Caps as the paper: max_fork 20, dw in [-5, 20]. One JSON line per cell. Run on box90 only.
"""
import json
import sys
import time

from pcmdp.howard import solve_average
from pcmdp.subchains import SubchainsModel

out, ratio, gamma, alphas = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
for alpha in alphas:
    t0 = time.time()
    _, g, it = solve_average(SubchainsModel(float(alpha), gamma, ratio, 20, -5, 20))
    row = {"ratio": ratio, "gamma": gamma, "alpha": float(alpha), "revenue": g, "I": 1 - g,
           "iterations": it, "seconds": round(time.time() - t0, 1)}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
