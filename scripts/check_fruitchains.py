"""Oracle run: Aumayr's Fruitchains output.json vs FruitchainsModel + howard.solve_average.

usage: check_fruitchains.py ORACLE_JSON OUT_JSONL [key ...]   (keys alpha_gamma_window_b2f_maxFork; default all)
Appends one JSON line per key: ours, theirs, seconds, states, iterations.
"""
import json
import sys
import time

from pcmdp.fruitchains import FruitchainsModel
from pcmdp.howard import solve_average

oracle = json.load(open(sys.argv[1]))
keys = sys.argv[3:] or list(oracle)
for key in keys:
    alpha, gamma, window, b2f, mfl = (float(x) for x in key.split("_"))
    t0 = time.time()
    model = FruitchainsModel(alpha, int(gamma), int(mfl), int(window), b2f)
    policy, g, it = solve_average(model, policy0=model.build_honest_policy(), verbose=True)
    rec = {"key": key, "ours": g, "theirs": oracle[key], "diff": g - oracle[key],
           "seconds": round(time.time() - t0), "states": model.state_space.size, "iterations": it}
    print(json.dumps(rec), flush=True)
    with open(sys.argv[2], "a") as f:
        f.write(json.dumps(rec) + "\n")
