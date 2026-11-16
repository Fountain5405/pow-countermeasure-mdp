"""Tests for the Publish or Perish model (pcmdp/pop.py).

Needs scipy, networkx and pymdptoolbox-roi; run on a lane box with the BLAS threads capped.
Oracles: Zhang & Preneel 2019, Q(0.3901) = 0.5372 for PoP and NC (gamma 0), PT(PoP) = 0.25.
"""
import math

import pytest

pytest.importorskip("scipy")
pytest.importorskip("networkx")
pytest.importorskip("mdptoolbox")

from pcmdp.howard import solve_average  # noqa: E402
from pcmdp.nakamoto import NakamotoModel  # noqa: E402
from pcmdp.pop import PublishOrPerishModel  # noqa: E402
from pcmdp.reorgs import MAJORITY, MINORITY, reorg_profile  # noqa: E402


def test_transitions_sum_to_one_and_illegal_actions_end():
    model = PublishOrPerishModel(0.35, max_b=4)
    for i in range(model.state_space.size):
        state = model.state_space.index_to_element(i)
        for j in range(model.action_space.size):
            action = model.action_space.index_to_element(j)
            t = model.get_state_transitions(state, action)
            assert math.isclose(sum(t.probabilities.values()), 1, abs_tol=1e-12)
            for ns, tags in t.reorgs.items():
                assert sum(tags.values()) <= t.probabilities[ns] + 1e-12


def test_honest_policy_earns_alpha():
    alpha = 0.3
    model = PublishOrPerishModel(alpha, max_b=4)
    policy = [int(model.Action.Adopt)] * model.state_space.size
    policy[model.state_space.element_to_index((0, 1, 0, 0, 1, 0))] = int(model.Action.Override)
    profile = reorg_profile(model, tuple(policy))
    assert profile["revenue"] == pytest.approx(alpha, abs=1e-12)
    assert profile[f"{MAJORITY}_reorgs_per_block"] == 0.0


def test_override_tags_the_honest_blocks_lost():
    model = PublishOrPerishModel(0.3, max_b=6)
    # (Bh, Bs, Dw, luck, last, published) = (1, 3, 1, 0, S, 0): minDw = -1, publish 3 blocks
    t = model.get_state_transitions((1, 3, 1, 0, 1, 0), model.Action.Override)
    assert t.probabilities == pytest.approx({(0, 1, 0, 0, 1, 0): 0.3, (1, 0, 1, 0, 0, 0): 0.7})
    assert all(r == 3 for r in t.rewards.values())
    assert t.reorgs[(0, 1, 0, 0, 1, 0)] == {(MAJORITY, 1): 0.3}
    assert t.reorgs[(1, 0, 1, 0, 0, 0)] == {(MAJORITY, 1): 0.7}


def test_match_tags_the_tie_outcomes():
    a = 0.3
    model = PublishOrPerishModel(a, max_b=6)
    # (2, 2, 1, 0, H, 0): minDw = 0, publish 2 = match the honest weight
    t = model.get_state_transitions((2, 2, 1, 0, 0, 0), model.Action.Match)
    on_honest, on_attacker, own = (3, 2, 2, 0, 0, 2), (1, 0, 1, 0, 0, 0), (2, 3, 0, 1, 1, 2)
    assert t.probabilities == pytest.approx({own: a, on_honest: 0.35, on_attacker: 0.35})
    assert t.rewards[on_attacker] == 2 and t.rewards[on_honest] == 0
    assert t.reorgs[on_honest] == {(MINORITY, 2): pytest.approx(0.35)}
    assert t.reorgs[on_attacker] == {(MAJORITY, 2): pytest.approx(0.35)}
    assert own not in t.reorgs


def test_small_oracle_pop_below_nakamoto_and_honest_below_threshold():
    # max_b 8 pinned at 0.458233 (2026-11-16, box90); max_b 30 should approach NC's 0.4628
    _, g, _ = solve_average(PublishOrPerishModel(0.3901, max_b=8))
    assert g == pytest.approx(0.458233, abs=1e-5)
    _, nc, _ = solve_average(NakamotoModel(0.3901, 0.0, 80))
    assert nc == pytest.approx(0.4628, abs=5e-4)
    assert g < nc
    # below the PT (0.25) the optimum is honest mining
    _, low, _ = solve_average(PublishOrPerishModel(0.2, max_b=8))
    assert low == pytest.approx(0.2, abs=1e-4)
