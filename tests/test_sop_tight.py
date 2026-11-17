"""Tests for the tight SoP disruption model (sop_tight.py)."""
import os

import numpy as np
import pytest

pytest.importorskip("scipy")

from pcmdp.howard import solve_tables  # noqa: E402
from pcmdp.sop_tight import (GIVE_UP, HOLD, PUBLISH, REVEAL, TightSoP,  # noqa: E402
                                                   exact_rate, profile)
from pcmdp import sop_rules as M  # noqa: E402


def small():
    return TightSoP(0.45, min_depth=2, x_min=-20, x_max=20, p_max=20, q_max=20)


def test_index_round_trip_and_rows():
    m = small()
    P, r, d, legal = m.tables()
    idx = np.arange(m.n)
    assert all((np.asarray(a) == np.asarray(b)).all()
               for a, b in zip(m.decode(m.index(*m.decode(idx))), m.decode(idx)))
    for a in range(4):
        assert np.allclose(P[a].sum(axis=1).A.ravel(), 1.0)
    h, a1, x, pa, ph = m.decode(idx)
    assert legal[HOLD].all()
    assert (legal[REVEAL] == ((a1 == 1) & (h >= 1) & (x - pa > 0))).all()
    assert not legal[GIVE_UP][(h == 0) & (a1 == 0)].any()
    assert not legal[PUBLISH][h >= 1].any()


def test_restarts_follow_the_exact_game():
    m = small()
    P, _, _, _ = m.tables()
    s = (2, 3, 0, 4, 1, 0, 6)              # a, D, xa, pa, h, xh, ph (exact)
    i = m.from_exact(s)
    for act, fn in [(REVEAL, M.do_reveal), (GIVE_UP, M.do_give_up)]:
        j = P[act][i].indices[0]
        assert j == m.from_exact(fn(s)[0])


def test_howard_value_equals_profile():
    m = small()
    P, r, d, legal = m.tables()
    policy, g, _ = solve_tables(P, r, d, legal, m.start, cur=m.initial_policy())
    assert g > 0
    assert profile(P, r, d, policy, m.start)["rate"] == pytest.approx(g, rel=1e-6)


def test_exact_game_without_window_is_close_to_the_model():
    """The tight model differs from the exact game only by the window and its caps."""
    m = TightSoP(0.45, min_depth=2, x_min=-24, x_max=24, p_max=30, q_max=30)
    P, r, d, legal = m.tables()
    policy, g, _ = solve_tables(P, r, d, legal, m.start, cur=m.initial_policy())
    res = exact_rate(m, policy, 2_000_000, seed=3, window=False)
    assert res["deep"] > 1000
    assert res["rate"] == pytest.approx(g, rel=0.1)


def _g(own_pending):
    m = TightSoP(0.40, min_depth=2, x_min=-20, x_max=20, p_max=20, q_max=20, own_pending=own_pending)
    P, r, d, legal = m.tables()
    return solve_tables(P, r, d, legal, m.start, cur=m.initial_policy())[1]


def test_own_pending_count_only_adds_reveals():
    g_ignore, g_count = _g("ignore"), _g("count")
    assert g_count >= g_ignore
    assert g_ignore == pytest.approx(0.016424926786388225, rel=1e-9)   # the value before the switch existed


def test_count_reveal_drops_published_shares():
    m = TightSoP(0.40, min_depth=2, x_min=-20, x_max=20, p_max=20, q_max=20, own_pending="count")
    P, _, _, legal = m.tables()
    # exact: D=3, pa=4, ph=5 -> k = 3 published, 1 stays; tight x = D + pa - ph = 2
    s = (2, 3, 0, 4, 1, 0, 5)
    i = m.from_exact(s)
    assert legal[REVEAL][i]
    j = P[REVEAL][i].indices[0]
    assert j == m.from_exact(M.do_reveal(s, True)[0]) == m.index(0, 0, 1, 1, 0)
    # ignore: nothing published
    assert M.do_reveal(s, False)[0] == (0, 0, 0, 4, 0, 0, 0)


def test_count_after_fix_not_above_legacy():
    def g(legacy):
        m = TightSoP(0.40, min_depth=2, x_min=-20, x_max=20, p_max=20, q_max=20,
                     own_pending="count", legacy_carry=legacy)
        P, r, d, legal = m.tables()
        return solve_tables(P, r, d, legal, m.start, cur=m.initial_policy())[1]
    assert g(False) <= g(True) + 1e-12


@pytest.mark.skipif(not os.environ.get("PCMDP_SLOW"),
                    reason="~45 min on box90 (10 M exact-game objects); set PCMDP_SLOW=1")
def test_count_model_matches_exact_game():
    """alpha 0.40, min_depth 5, non-binding caps, exact game without the window, 10 M objects."""
    m = TightSoP(0.40, min_depth=5, x_min=-48, x_max=48, p_max=48, q_max=24, own_pending="count")
    P, r, d, legal = m.tables()
    policy, g, _ = solve_tables(P, r, d, legal, m.start, cur=m.initial_policy())
    res = exact_rate(m, policy, 10_000_000, seed=5, window=False)
    n = res["deep"]
    assert n > 100
    sigma = g * n ** -0.5       # Poisson count error on the rate
    assert abs(res["rate"] - g) < 3 * sigma
