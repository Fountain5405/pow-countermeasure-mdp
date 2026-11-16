"""Solve RS / PRS grid points and compare with the reference's output.json.

  python scripts/check_reward_splitting.py RS|PRS REF_OUTPUT_JSON OUT_JSONL [KEY ...]

KEY = alpha_gamma_window_forkWindow_maxForkLen as in the reference; no KEY = every
key in REF_OUTPUT_JSON. Keys already in OUT_JSONL are skipped. One JSON line per key:
{"key", "ours", "ref", "diff", "seconds", "states", "iterations"}. Run on box90 only.
"""
import json
import sys
import time

from pcmdp.howard import solve_average
from pcmdp.reward_splitting import ProportionalRewardSplittingModel, RewardSplittingModel

kind, ref_path, out_path, keys = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
ref = json.load(open(ref_path))
cls = {"RS": RewardSplittingModel, "PRS": ProportionalRewardSplittingModel}[kind]
try:
    done = {json.loads(line)["key"] for line in open(out_path)}
except FileNotFoundError:
    done = set()
for key in keys or list(ref):
    if key in done:
        continue
    alpha, gamma, window, fork_window, cap = (float(x) for x in key.split("_"))
    t0 = time.time()
    model = cls(alpha, gamma, int(cap), int(window), int(fork_window))
    _, g, it = solve_average(model)
    row = {"key": key, "ours": g, "ref": ref[key], "diff": g - ref[key],
           "seconds": round(time.time() - t0, 1), "states": model.state_space.size, "iterations": it}
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)
