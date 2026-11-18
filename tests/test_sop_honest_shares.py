"""sop_tight / sop_rules with honest_shares: MRL #146's slot rule on the fork point, level restarts."""
import numpy as np
import pytest

from pcmdp import sop_rules as M
from pcmdp.howard import solve_tables
from pcmdp.sop_tight import GIVE_UP, HOLD, REVEAL, TightSoP, exact_rate, profile


def model(**kw):
    base = dict(alpha=0.45, min_depth=3, x_min=-24, x_max=24, p_max=16, q_max=16, own_pending="count")
    base.update(kw)
    return TightSoP(**base)


def hold_successors(m, P, state):
    row = P[HOLD][m.index(*state)]
    return {tuple(int(v) for v in m.decode(j)): p for j, p in zip(row.indices, row.data)}


@pytest.mark.parametrize("honest_shares", [False, True])
def test_honest_share_on_the_fork_point(honest_shares):
    m = model(honest_shares=honest_shares)
    P, r, d, legal = m.tables()
    q = (1 - m.alpha) * (1 - 1 / m.w)
    # level, no blocks: the honest share's slot is free on the attacker's side
    nxt = hold_successors(m, P, (0, 0, 0, 0, 0))
    want = (0, 0, 0, 1, 1) if honest_shares else (0, 0, -1, 0, 1)
    assert nxt[want] == pytest.approx(q)
    # the attacker already holds that slot (pa 2 > ph 1): the share counts for honest only
    assert hold_successors(m, P, (0, 0, 1, 2, 1))[(0, 0, 0, 2, 2)] == pytest.approx(q)
    # after the attacker's first block the honest shares fit nothing of the attacker's
    assert hold_successors(m, P, (0, 1, 0, 0, 0))[(0, 1, -1, 0, 1)] == pytest.approx(q)
    # both counts at their caps (16): the slot rule reads x (= pa - ph with no block), not the caps
    want = (0, 0, 2, 16, 16) if honest_shares else (0, 0, 2, 16, 16)
    assert hold_successors(m, P, (0, 0, 3, 16, 16))[want] == pytest.approx(q)


def test_publish_drops_the_old_fork_points_honest_shares():
    m = model(honest_shares=True)
    P, r, d, legal = m.tables()
    row = P[3][m.index(0, 1, 2, 5, 4)]          # PUBLISH
    assert tuple(int(v) for v in m.decode(row.indices[0])) == (0, 0, 5, 5, 0)
    assert M.do_publish((2, 7, 0, 5, 0, 0, 4), honest_shares=True)[0] == (0, 0, 0, 5, 0, 0, 0)
    assert M.do_publish((2, 7, 0, 5, 0, 0, 4))[0] == (0, 0, 0, 5, 0, 0, 4)        # Model B as built


@pytest.mark.parametrize("honest_shares", [False, True])
def test_give_up_restarts_level(honest_shares):
    m = model(honest_shares=honest_shares)
    P, r, d, legal = m.tables()
    row = P[GIVE_UP][m.index(2, 1, -5, 3, 4)]
    got = tuple(int(v) for v in m.decode(row.indices[0]))
    assert got == ((0, 0, 0, 4, 4) if honest_shares else (0, 0, -4, 0, 4))


def test_exact_rules_follow_the_slot_rule():
    caps = M.Caps(p_max=100)
    s = M.event(M.START, False, False, caps, honest_shares=True)
    assert s == (0, 0, 0, 1, 0, 0, 1)
    s = M.event(s, True, False, caps, honest_shares=True)         # attacker share: pa 2, ph 1
    s = M.event(s, False, False, caps, honest_shares=True)        # slot 1 taken: honest only
    assert s == (0, 0, 0, 2, 0, 0, 2)
    s = M.event(s, False, True, caps, honest_shares=True)         # honest block
    s = M.event(s, False, False, caps, honest_shares=True)        # after a block: honest only
    assert s[3] == 2 and s[6] == 1
    assert M.do_give_up((1, 3, 0, 2, 2, 0, 5), honest_shares=True)[0] == (0, 0, 0, 5, 0, 0, 5)
    assert M.event(M.START, False, False, caps) == (0, 0, 0, 0, 0, 0, 1)   # default unchanged


def test_honest_shares_help_the_attacker():
    rates = {}
    for hs in (False, True):
        m = model(honest_shares=hs)
        P, r, d, legal = m.tables()
        _, g, _ = solve_tables(P, r, d, legal, m.start, cur=m.initial_policy())
        rates[hs] = g
    assert rates[True] > rates[False]


def test_model_matches_exact_game_with_honest_shares():
    # the optimal policy run through the exact game (no window, as the model): agree within 4 sigma
    m = model(honest_shares=True, x_min=-40, x_max=40, p_max=24, q_max=24)
    P, r, d, legal = m.tables()
    policy, g, _ = solve_tables(P, r, d, legal, m.start, cur=m.initial_policy())
    res = exact_rate(m, policy, 2_000_000, seed=3, window=False)
    sigma = np.sqrt(res["deep"]) / (res["objects"] / m.w)
    assert res["deep"] > 1000
    assert abs(res["rate"] - g) < 4 * sigma + 0.01 * g


def test_stubborn_policy_is_legal_and_rate_positive():
    m = model(honest_shares=True, publish_all_on_reveal=True)
    P, r, d, legal = m.tables()
    h, a1, x, pa, ph = m.decode(np.arange(m.n))
    pol = np.full(m.n, HOLD)
    pol[((h >= 1) | (a1 == 1)) & (x < -10)] = GIVE_UP
    pol[(h >= m.min_depth) & (x == 0)] = GIVE_UP
    pol[(h >= m.min_depth) & (x > 0) & (a1 == 1)] = REVEAL
    assert legal[pol, np.arange(m.n)].all()
    assert profile(P, r, d, pol, m.start)["rate"] > 0
