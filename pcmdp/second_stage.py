"""Second stage: the deep reorgs of the revenue-optimal attacker, from fewest to most.

Many policies earn the optimal revenue, and they can differ by orders of
magnitude in the deep reorgs they cause (Nakamoto, alpha 0.30, gamma 0.5:
PTO's optimal policy 1.4e-6 per main-chain block, Howard's 5.2e-4, at the
same revenue). This module reports the least and the most.

1. Evaluate the revenue-optimal policy (from `solve --save-policy`) for its
   gain g* and bias h*, and check that no action improves on it. In every
   state keep the actions with Q(s, a) = r - g* d + P h* within eps of the
   best. With the Dinkelbach reward r - g* d the optimal gain is 0, and for
   any policy pi with stationary distribution mu,
   mu (r_pi - g* d_pi) = -sum_s mu(s) (max_a Q(s, a) - Q(s, pi(s))).
   So a policy is revenue-optimal if and only if it uses only these actions
   in the states it visits; its actions elsewhere do not change its rates.
2. On the same model with the disruption reward (majority reorgs of
   min_depth+ blocks per block interval), run Howard over those actions
   only, once maximising the rate and once minimising it, each starting from
   the revenue-optimal policy.

Each of the three policies (Howard's revenue-optimal one, the minimum and
the maximum) gets a reorg profile. Their revenues must agree to about eps.

    python -m pcmdp.second_stage pol/nakamoto_a0.30_g0.5_rev.npz [--eps 1e-9]

The .npz is one `solve.py --save-policy` wrote for a revenue job; the model
is rebuilt from the arguments saved with it. One JSON line on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from scipy.sparse import diags

from .howard import _tables, evaluate, solve_tables
from .reorgs import reorg_profile


def policy_tables(P, cur):
    na = len(P)
    return sum(diags((cur == a).astype(float)) @ P[a] for a in range(na)).tocsr()


def optimal_actions(P, r, d, legal, start, cur, eps: float):
    """(g*, mask of revenue-optimal actions, the largest improvement any action offers)."""
    n = r.shape[1]
    rows = np.arange(n)
    g, h = evaluate(policy_tables(P, cur), r[cur, rows], d[cur, rows], start)
    Q = np.full(r.shape, -np.inf)
    for a in range(r.shape[0]):
        Q[a] = np.where(legal[a], r[a] - g * d[a] + P[a] @ h, -np.inf)
    best = Q.max(axis=0)
    gap = float(np.max(best - Q[cur, rows]))
    keep = Q >= (best - eps * np.maximum(1.0, np.abs(best)))[None, :]
    return g, keep & legal, gap


def same_chain(P1, P2) -> bool:
    if len(P1) != len(P2):
        return False
    for a, b in zip(P1, P2):
        if a.shape != b.shape or a.nnz != b.nnz or abs(a - b).max() > 1e-12:
            return False
    return True


def run(rev_model, dis_model, policy, eps: float = 1e-9, max_iter: int = 200, verbose: bool = False) -> dict:
    """Min and max deep-reorg rate over the revenue-optimal policies of rev_model.

    `policy` is a revenue-optimal policy over all states (final included), as
    solve.py saves it. dis_model must be the same protocol and caps with the
    disruption objective. Returns a dict of the three profiles and checks.
    """
    t0 = time.time()
    mdp, keep, P, r, d, legal, start = _tables(rev_model)
    cur = np.asarray(policy)[keep]
    g, allowed, gap = optimal_actions(P, r, d, legal, start, cur, eps)
    if gap > 1e-9:
        raise ValueError(f"the policy is not revenue-optimal: an action improves it by {gap:.3g}")
    allowed[cur, np.arange(len(cur))] = True   # its own actions, even if a hair below eps
    tied = int((allowed.sum(axis=0) > 1).sum())
    out = {"eps": eps, "revenue_g": g, "improvement_gap": gap, "tied_states": tied,
           "states": int(len(keep)), "stage1_s": round(time.time() - t0, 1)}
    del r, d
    t0 = time.time()
    mdp2, keep2, P2, r2, d2, legal2, start2 = _tables(dis_model)
    if not (np.array_equal(keep, keep2) and start == start2 and same_chain(P, P2)):
        raise ValueError("the revenue and disruption models do not have the same chain")
    del P
    allowed &= legal2
    full = np.zeros(mdp2.num_of_states, dtype=int)
    for tag, sign in (("max", 1.0), ("min", -1.0)):
        pol, val, it = solve_tables(P2, sign * r2, d2, allowed, start2, cur=cur, max_iter=max_iter, verbose=verbose)
        full[keep2] = pol
        out[tag] = {"g": sign * val, "iterations": it, "changed_states": int((pol != cur).sum()),
                    "policy": full.copy()}
    out["stage2_s"] = round(time.time() - t0, 1)
    full[keep2] = cur
    out["howard"] = {"policy": full.copy()}
    t0 = time.time()
    for tag in ("howard", "min", "max"):
        pol = tuple(int(x) for x in out[tag].pop("policy"))
        out[tag]["profile"] = reorg_profile(dis_model, pol, max_report_depth=40)
    out["profile_s"] = round(time.time() - t0, 1)
    revs = [out[t]["profile"]["revenue"] for t in ("howard", "min", "max")]
    out["revenue_spread"] = max(revs) - min(revs)
    return out


def main(argv=None) -> int:
    import contextlib

    from .solve import build_model

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy_npz", help="a revenue policy saved by solve.py --save-policy")
    p.add_argument("--eps", type=float, default=1e-9,
                   help="keep actions whose Q is within eps (relative, floor 1) of the best")
    p.add_argument("--min-depth", type=int, default=11)
    p.add_argument("--max-iter", type=int, default=200)
    args = p.parse_args(argv)

    saved = np.load(args.policy_npz)
    sargs = argparse.Namespace(**json.loads(str(saved["args"])))
    if sargs.objective != "revenue":
        p.error("the saved policy is not a revenue policy")
    if sargs.protocol == "sop":
        p.error("the SoP walk model has no revenue objective")
    rev_model = build_model(sargs)
    dargs = argparse.Namespace(**vars(sargs))
    dargs.objective, dargs.normalization, dargs.weight, dargs.min_depth = (
        "disruption", "time", "count", args.min_depth)
    dis_model = build_model(dargs)
    policy = tuple(int(x) for x in saved["policy"])
    if len(policy) != rev_model.state_space.size:
        raise ValueError("the saved policy does not fit this model")
    out = {"protocol": sargs.protocol, "alpha": sargs.alpha, "gamma": sargs.gamma, "max_fork": sargs.max_fork,
           "model": repr(rev_model), "normalization_revenue": sargs.normalization, "min_depth": args.min_depth,
           "unit": "block interval", "policy_npz": args.policy_npz}
    with contextlib.redirect_stdout(sys.stderr):   # Howard's progress lines; stdout carries the JSON
        out.update(run(rev_model, dis_model, policy, eps=args.eps, max_iter=args.max_iter, verbose=True))
    json.dump(out, sys.stdout, default=float)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
