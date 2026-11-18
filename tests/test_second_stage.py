"""pcmdp.second_stage: fewest and most deep reorgs over the revenue-optimal policies."""
import itertools

import numpy as np
import pytest

from pcmdp.howard import _tables, evaluate, solve_average
from pcmdp.nakamoto import NakamotoModel
from pcmdp.second_stage import optimal_actions, policy_tables, run


def models(alpha, gamma, max_fork, min_depth):
    rev = NakamotoModel(alpha, gamma, max_fork)
    dis = NakamotoModel(alpha, gamma, max_fork, objective="disruption", min_depth=min_depth,
                        normalization="time")
    return rev, dis


def brute_force(rev, dis, policy, eps, limit=14):
    """Every policy that differs from `policy` only by revenue-optimal actions in the tied
    states: (min, max) disruption rate over those whose chain the evaluation can solve,
    and the largest revenue difference to g*."""
    _, keep, P, r, d, legal, start = _tables(rev)
    cur = np.asarray(policy)[keep]
    rows = np.arange(len(cur))
    g, allowed, _ = optimal_actions(P, r, d, legal, start, cur, eps)
    allowed[cur, rows] = True
    _, _, P2, r2, d2, _, _ = _tables(dis)
    tied = np.nonzero(allowed.sum(axis=0) > 1)[0]
    if len(tied) > limit:
        pytest.skip(f"{len(tied)} tied states: too many to enumerate")
    rates, worst = [], 0.0
    for combo in itertools.product(*[np.nonzero(allowed[:, s])[0] for s in tied]):
        pol = cur.copy()
        pol[tied] = combo
        try:
            gd, _ = evaluate(policy_tables(P2, pol), r2[pol, rows], d2[pol, rows], start)
            gr, _ = evaluate(policy_tables(P, pol), r[pol, rows], d[pol, rows], start)
        except RuntimeError:   # a closed class away from the start state: not unichain
            continue
        rates.append(gd)
        worst = max(worst, abs(gr - g))
    return min(rates), max(rates), worst, len(tied)


# (0.30, 0.5, 8, 6): 7 tied states, min 4.10e-5, max 8.20e-5 per block interval;
# (0.30, 0.5, 8, 3): 7 tied states, and the ties do not change the rate
@pytest.mark.parametrize("alpha,gamma,max_fork,min_depth", [(0.30, 0.5, 8, 6), (0.30, 0.5, 8, 3)])
def test_bounds_match_brute_force(alpha, gamma, max_fork, min_depth):
    rev, dis = models(alpha, gamma, max_fork, min_depth)
    policy, g, _ = solve_average(rev)
    out = run(rev, dis, policy, eps=1e-9)
    lo, hi, worst, n = brute_force(rev, dis, policy, 1e-9)
    assert out["tied_states"] == n
    assert worst < 1e-9                       # every enumerated policy earns g*
    assert out["min"]["g"] == pytest.approx(lo, rel=1e-9, abs=1e-15)
    assert out["max"]["g"] == pytest.approx(hi, rel=1e-9, abs=1e-15)


def test_ties_matter_at_deep_reorgs():
    # cap 8, depth 6: the revenue-optimal policies span a factor 2
    rev, dis = models(0.30, 0.5, 8, 6)
    out = run(rev, dis, solve_average(rev)[0])
    assert out["max"]["g"] == pytest.approx(2 * out["min"]["g"], rel=1e-3)


# cap 40, depth 11: min 1.07e-6 and max 4.29e-4 at (0.30, 0.5); 0 and 4.46e-3 at (0.35, 1)
@pytest.mark.parametrize("alpha,gamma", [(0.30, 0.5), (0.35, 1.0), (0.40, 0.0)])
def test_revenue_kept_and_rates_ordered(alpha, gamma):
    rev, dis = models(alpha, gamma, 40, 11)
    policy, g, _ = solve_average(rev)
    out = run(rev, dis, policy, eps=1e-9)
    assert out["revenue_g"] == pytest.approx(g, abs=1e-12)
    assert out["revenue_spread"] < 1e-8
    deep = {t: out[t]["profile"]["majority_deep_reorgs_per_block"] for t in ("min", "howard", "max")}
    assert deep["min"] <= deep["howard"] * (1 + 1e-9) + 1e-15
    assert deep["howard"] <= deep["max"] * (1 + 1e-9) + 1e-15
    # each stage-2 value is the rate its own profile reports
    for t in ("min", "max"):
        assert out[t]["g"] == pytest.approx(deep[t], rel=1e-7, abs=1e-15)


def test_rejects_a_policy_that_is_not_revenue_optimal():
    rev, dis = models(0.40, 0.0, 12, 3)
    honest = rev.build_honest_policy()
    with pytest.raises(ValueError, match="not revenue-optimal"):
        run(rev, dis, honest)
