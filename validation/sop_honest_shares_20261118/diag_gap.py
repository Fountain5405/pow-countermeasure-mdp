"""Model g against the exact game for the model's optimal policy, alpha 0.45, depth 3, by variant."""
import sys

import numpy as np

from pcmdp.howard import solve_tables
from pcmdp.sop_tight import TightSoP, exact_rate

cases = {
    "count": dict(honest_shares=False, x_min=-40, x_max=40, p_max=24, q_max=24),
    "count_hs": dict(honest_shares=True, x_min=-40, x_max=40, p_max=24, q_max=24),
    "count_hs_big": dict(honest_shares=True, x_min=-64, x_max=64, p_max=48, q_max=48),
}
for name in sys.argv[1:] or cases:
    m = TightSoP(alpha=0.45, min_depth=3, own_pending="count", **cases[name])
    P, r, d, legal = m.tables()
    policy, g, _ = solve_tables(P, r, d, legal, m.start, cur=m.initial_policy())
    res = exact_rate(m, policy, 2_000_000, seed=3, window=False)
    sigma = np.sqrt(res["deep"]) / (res["objects"] / m.w)
    print(f"{name}: states {m.n} model {g:.5f} exact {res['rate']:.5f} +- {sigma:.5f} "
          f"({(res['rate'] - g) / sigma:+.1f} sigma)", flush=True)
