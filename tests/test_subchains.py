"""Tests for the Subchains incentive-compatibility MDP. Run on box90 (see docs/PORTING.md)."""
import math

import pytest

pytest.importorskip("scipy")
pytest.importorskip("networkx")
pytest.importorskip("mdptoolbox")

from pcmdp.howard import solve_average  # noqa: E402
from pcmdp.nakamoto import NakamotoModel  # noqa: E402
from pcmdp.reorgs import MAJORITY, MINORITY, reorg_profile  # noqa: E402
from pcmdp.subchains import SubchainsModel  # noqa: E402

SMALL = dict(max_fork=6, min_dw=-3, max_dw=8)


def small(alpha=0.3, gamma=0.5, ratio=2):
    return SubchainsModel(alpha, gamma, ratio, **SMALL)


def honest_policy(model):
    """Override whenever the attacker's chain weighs more, else adopt."""
    A = model.Action
    policy = []
    for i in range(model.state_space.size):
        state = model.state_space.index_to_element(i)
        action = A.Override if state[2] > 0 else A.Adopt
        policy.append(model.action_space.element_to_index(action))
    return tuple(policy)


def test_probabilities_sum_to_one():
    model = small()
    for i in range(model.state_space.size):
        state = model.state_space.index_to_element(i)
        for j in range(model.action_space.size):
            action = model.action_space.index_to_element(j)
            t = model.get_state_transitions(state, action)
            assert math.isclose(sum(t.probabilities.values()), 1, abs_tol=1e-12)
            for nxt, tags in t.reorgs.items():
                assert sum(tags.values()) <= t.probabilities[nxt] + 1e-12


@pytest.mark.parametrize("ratio", [1, 2, 3])
def test_honest_policy_earns_alpha(ratio):
    model = small(0.3, 0.5, ratio)
    profile = reorg_profile(model, honest_policy(model))
    assert profile["revenue"] == pytest.approx(0.3, abs=1e-9)
    assert profile["majority_reorgs_per_block"] == 0


def test_weak_attacker_still_profits():
    # Result 9 of the paper: the profitable threshold is 0, so unlike Nakamoto
    # the optimum at alpha 0.1 is above alpha (Table V: I = 0.8990 at gamma 0)
    model = small(0.1, 0.0, 2)
    _, g, _ = solve_average(model)
    assert g > 0.1 + 1e-4
    assert g == pytest.approx(1 - 0.8990, abs=2e-3)


def test_table_v_oracle_gamma_one_closed_form():
    # Table V, gamma = 1: I = (1 - 2 alpha) / (1 - alpha), i.e. revenue alpha / (1 - alpha)
    _, g, _ = solve_average(small(0.2, 1.0, 2))
    assert g == pytest.approx(0.2 / 0.8, abs=2e-3)


def test_ratio_one_is_below_nakamoto_and_equal_when_honest_is_best():
    # one unit per block = Nakamoto, except the reference keeps at most three
    # private units after a publication and matches only up to dw 3 (compliant-
    # favouring truncation), so it can only lose revenue
    _, g1, _ = solve_average(SubchainsModel(0.1, 0.5, 1, **SMALL))
    assert g1 == pytest.approx(0.1, abs=1e-9)
    for alpha, gamma in [(0.4, 0.5), (0.35, 1.0)]:
        _, sub, _ = solve_average(SubchainsModel(alpha, gamma, 1, 12, -5, 20))
        _, nak, _ = solve_average(NakamotoModel(alpha, gamma, 12))
        assert nak - 0.04 < sub <= nak + 1e-9


def tags(model, state, action):
    return model.get_state_transitions(state, model.Action[action]).reorgs


def test_reorg_tags_hand_checked():
    m = small(0.3, 0.5, 2)
    F = m.Fork
    # override with dw 2, h 1, a 2 (last = 11): keeps one unit (a block);
    # honest chain loses h = 1 block on every successor
    t = m.get_state_transitions((2, 1, 2, F.Irrelevant, 3), m.Action.Override)
    assert all(v == {(MAJORITY, 1): pytest.approx(t.probabilities[n])} for n, v in t.reorgs.items())
    assert len(t.reorgs) == 4
    # Match from Relevant, a = 3 blocks, last = 0b101 at dw 3 -> blocksInLast 2,
    # published 1 block; honest on the attacker's branch: majority loses h = 2
    t = m.get_state_transitions((3, 2, 3, F.Relevant, 5), m.Action.Match)
    # honest block on the honest branch: minority loses the 1 published block
    nxt = (3, 3, 2, F.Relevant, 1)
    assert t.reorgs[nxt] == {(MINORITY, 1): pytest.approx(t.probabilities[nxt])}
    # honest block on the attacker's branch: a = blocksInLast = 2, h = 1
    nxt = (2, 1, 2, F.Relevant, 1)
    assert t.reorgs[nxt] == {(MAJORITY, 2): pytest.approx(t.probabilities[nxt])}
    assert t.rewards[nxt] == 1
    # no minority tag without gamma
    m0 = small(0.3, 0.0, 2)
    t = m0.get_state_transitions((3, 2, 3, F.Relevant, 5), m0.Action.Match)
    assert not any(k == MINORITY for v in t.reorgs.values() for (k, _) in v)
