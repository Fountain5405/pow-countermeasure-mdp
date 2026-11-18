"""Under own_pending="count" with honest_shares, ph (q_max) should not change the optimum."""
from pcmdp.howard import solve_tables
from pcmdp.sop_tight import TightSoP

for alpha, depth, xr in [(0.45, 3, 40), (0.40, 6, 40), (0.40, 11, 48)]:
    gs = {}
    for q in (0, 8, 24):
        m = TightSoP(alpha, min_depth=depth, x_min=-xr, x_max=xr, p_max=24, q_max=q, own_pending="count",
                     honest_shares=True)
        P, r, d, legal = m.tables()
        _, g, it = solve_tables(P, r, d, legal, m.start, cur=m.initial_policy())
        gs[q] = g
    print(f"alpha {alpha} depth {depth}: " + ", ".join(f"q {q}: {g:.12e}" for q, g in gs.items())
          + f"; max rel diff {max(abs(g - gs[0]) for g in gs.values()) / gs[0]:.1e}", flush=True)
