"""Tests for the Fruitchains incentive-compatibility model (fruitchains.py)."""
import math

import pytest

pytest.importorskip("scipy")

from pcmdp.fruitchains import FruitchainsModel  # noqa: E402
from pcmdp.howard import solve_average  # noqa: E402
from pcmdp.reorgs import MAJORITY, reorg_profile  # noqa: E402

SMALL = dict(max_fork=5, window=3, max_afa=4, max_hf=6)


def small(alpha=0.4, gamma=0, **kw):
    return FruitchainsModel(alpha, gamma, **{**SMALL, **kw})


@pytest.mark.parametrize("gamma", [0, 1])
def test_probabilities_sum_to_one_and_space_round_trips(gamma):
    m = small(gamma=gamma, block_to_fruit=2.0)
    for i in range(m.state_space.size):
        s = m.state_space.index_to_element(i)
        assert m.state_space.element_to_index(s) == i
        for a in range(m.action_space.size):
            t = m.get_state_transitions(s, m.action_space.index_to_element(a))
            assert math.isclose(sum(t.probabilities.values()), 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("gamma", [0, 1])
def test_honest_policy_earns_alpha(gamma):
    m = small(0.3, gamma, block_to_fruit=2.0)
    prof = reorg_profile(m, m.build_honest_policy())
    assert prof["revenue"] == pytest.approx(0.3, abs=1e-9)
    assert prof["majority_reorgs_per_block"] == 0


def test_optimum_beats_honest_and_matches_profile():
    m = small(0.45, 1)
    policy, g, _ = solve_average(m)
    assert g > 0.45
    assert reorg_profile(m, policy)["revenue"] == pytest.approx(g, rel=1e-9)


def test_override_tags_the_orphaned_honest_blocks():
    m = small(0.4, 0)
    t = m.get_state_transitions((3, 2, 0, 1, 0), m.Action.Override)
    assert sum(t.probabilities.values()) == pytest.approx(1)
    for nxt, p in t.probabilities.items():
        assert t.reorgs[nxt] == {(MAJORITY, 2): pytest.approx(p)}
        assert t.rewards[nxt] in (0, 1)         # afA = 0, +1 on the attacker-fruit outcome
    # gamma = 1 tie after an honest block: (2, 2, last=1) overrides, depth 2; gamma 0 cannot
    g1 = small(0.4, 1)
    t = g1.get_state_transitions((2, 2, 0, 0, 1), g1.Action.Override)
    assert t.reorgs and all(v == {(MAJORITY, 2): pytest.approx(t.probabilities[n])}
                            for n, v in t.reorgs.items())
    t = m.get_state_transitions((2, 2, 0, 0, 1), m.Action.Override)
    assert list(t.probabilities) == [m.final_state] and not t.reorgs
    # no honest blocks to orphan: no tag
    assert not m.get_state_transitions((1, 0, 0, 0, 0), m.Action.Override).reorgs


def test_adopt_credits_honest_fruits_and_is_tag_free():
    m = small(0.4, 0)
    t = m.get_state_transitions((2, 3, 0, 4, 1), m.Action.Adopt)
    assert not t.reorgs
    assert t.difficulty_contributions[(0, 0, 0, 0, 0)] == 5   # 4 honest fruits + the attacker fruit
    assert t.rewards[(0, 0, 0, 0, 0)] == 1


def test_oracle_point_reduced_caps():
    """Aumayr's key 0.25_0.0_6.0_1.0_30.0 = 0.26897 (full caps, scripts/check_fruitchains.py);
    here max_fork 12 / max_afa 12 / max_hf 20 must already be within 5e-3 of it and above honest."""
    theirs = 0.26897430419921875
    m = FruitchainsModel(0.25, 0, max_fork=12, window=6, max_afa=12, max_hf=20)
    _, g, _ = solve_average(m)
    assert g == pytest.approx(theirs, abs=5e-3)
    assert g > 0.25
