"""tevador's stubborn attacker, evaluated exactly in the tight model, against his simulator.

His attacker (MRL #146 appendix, tevador_sim.py; GAMMA 1 = ties to the honest
chain, COUNT_TIP_SHARES, depth 11): it holds; gives up (restarting level, with
the honest tip's shares) once the honest side leads by more than STUBBORNNESS
objects; when the honest side has >= 11 blocks and it is not behind, it
reveals if strictly ahead (publishing every pending share) and otherwise gives
up. In the tight model with honest_shares and publish_all_on_reveal that is a
fixed policy; its rate comes from the stationary distribution (no sampling).

    python stubborn_check.py <alpha> [--stubbornness 10] [--x-min -16] [--x-max 96] [--p-max 64] [--q-max 48]

His simulator's rates (2026-11-18, box90 and tr, block interval = mean time to
any block): alpha 0.45 4.37e-3 +- 8e-6; 0.40 1.16e-4 +- 1.7e-6; 1/3
2.76e-7 +- 1.7e-8 (with box90's 16 seeds 2.75e-7).
"""
import argparse
import json
import sys
import time

import numpy as np

from pcmdp.sop_tight import GIVE_UP, HOLD, REVEAL, TightSoP, profile


def stubborn_policy(m: TightSoP, s: int) -> np.ndarray:
    h, a1, x, pa, ph = m.decode(np.arange(m.n))
    pol = np.full(m.n, HOLD)
    pol[((h >= 1) | (a1 == 1)) & (x < -s)] = GIVE_UP
    pol[(h >= m.min_depth) & (x == 0)] = GIVE_UP
    pol[(h >= m.min_depth) & (x > 0) & (a1 == 1)] = REVEAL
    return pol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("alpha", type=float)
    p.add_argument("--stubbornness", type=int, default=10)
    p.add_argument("--min-depth", type=int, default=11)
    p.add_argument("--x-min", type=int, default=-16)
    p.add_argument("--x-max", type=int, default=96)
    p.add_argument("--p-max", type=int, default=64)
    p.add_argument("--q-max", type=int, default=48)
    a = p.parse_args()
    t0 = time.time()
    m = TightSoP(a.alpha, min_depth=a.min_depth, x_min=a.x_min, x_max=a.x_max, p_max=a.p_max, q_max=a.q_max,
                 own_pending="count", honest_shares=True, publish_all_on_reveal=True)
    P, r, d, legal = m.tables()
    pol = stubborn_policy(m, a.stubbornness)
    assert legal[pol, np.arange(m.n)].all()
    out = {"alpha": a.alpha, **vars(a), "states": m.n, **profile(P, r, d, pol, m.start),
           "s": round(time.time() - t0, 1)}
    json.dump(out, sys.stdout)
    print()


if __name__ == "__main__":
    main()
