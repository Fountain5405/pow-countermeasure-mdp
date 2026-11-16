"""Tests for the tie-breaking MDPs (UTB, UDTB, SHTB). Run on a lane box (box90), BLAS threads capped."""
import math

import pytest

pytest.importorskip("scipy")
pytest.importorskip("networkx")
pytest.importorskip("mdptoolbox")

from pcmdp.howard import solve_average  # noqa: E402
from pcmdp.nakamoto import NakamotoModel  # noqa: E402
from pcmdp.reorgs import MAJORITY, MINORITY, reorg_profile  # noqa: E402
from pcmdp.tie_breaking import (SmallestHashTieBreakingModel, UDTieBreakingModel,  # noqa: E402
                                UniformTieBreakingModel, _CompactSpace)


def small_models(alpha=0.3):
    return [UniformTieBreakingModel(alpha, 6), UDTieBreakingModel(alpha, 6),
            SmallestHashTieBreakingModel(alpha, 5, 3)]


@pytest.mark.parametrize("model", small_models(), ids=lambda m: type(m).__name__)
def test_probabilities_sum_to_one_and_states_canonical(model):
    for i in range(model.state_space.size):
        state = model.state_space.index_to_element(i)
        assert model.state_space.element_to_index(state) == i
        for j in range(model.action_space.size):
            t = model.get_state_transitions(state, model.action_space.index_to_element(j))
            assert math.isclose(sum(t.probabilities.values()), 1, abs_tol=1e-12)
            for nxt, tags in t.reorgs.items():
                assert model.state_space.index_to_element(model.state_space.element_to_index(nxt)) == nxt
                assert sum(tags.values()) <= t.probabilities[nxt] * 2 + 1e-12


def test_compact_space_is_a_bijection():
    s = _CompactSpace(7, 4)
    seen = {s.index_to_element(i) for i in range(s.size)}
    assert len(seen) == s.size
    assert all(s.element_to_index(e) == i for i, e in enumerate(sorted(seen, key=s.element_to_index)))


@pytest.mark.parametrize("model", small_models(), ids=lambda m: type(m).__name__)
def test_honest_policy_earns_alpha(model):
    assert reorg_profile(model, model.build_honest_policy())["revenue"] == pytest.approx(model.alpha, abs=1e-9)


def test_honest_is_optimal_below_threshold():
    for model in (UniformTieBreakingModel(0.1, 10), UDTieBreakingModel(0.1, 10)):
        assert solve_average(model)[1] == pytest.approx(0.1, abs=1e-9)
    assert solve_average(UniformTieBreakingModel(0.3, 10))[1] > 0.33
    assert solve_average(UDTieBreakingModel(0.3, 10))[1] > 0.33


def test_shtb_with_one_region_is_nakamoto_gamma_zero():
    # one region: the public tip always wins ties, the attacker never does
    for alpha in (0.2, 0.3, 0.4):
        ours = solve_average(SmallestHashTieBreakingModel(alpha, 10, 1))[1]
        theirs = solve_average(NakamotoModel(alpha, 0.0, 10))[1]
        assert ours == pytest.approx(theirs, abs=1e-9)


def bisect_pt(make, lo, hi):
    while hi - lo > 1e-4:
        mid = (lo + hi) / 2
        if solve_average(make(mid))[1] > mid + 1e-4:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def test_profitable_threshold_oracle():
    # Zhang & Preneel Table II: 0.2321 for UTB and UDTB (here lmax 40)
    assert bisect_pt(lambda a: UniformTieBreakingModel(a, 40), 0.22, 0.25) == pytest.approx(0.2321, abs=0.002)
    assert bisect_pt(lambda a: UDTieBreakingModel(a, 40), 0.22, 0.25) == pytest.approx(0.2321, abs=0.002)


def tags(model, state, action):
    t = model.get_state_transitions(state, action)
    return {s: (p, t.reorgs.get(s, {})) for s, p in t.probabilities.items()}


def test_utb_tags():
    m = UniformTieBreakingModel(0.3, 6)
    got = tags(m, (3, 2, 0), m.Action.Match)
    assert got[(4, 2, 1)][1] == {}
    assert got[(1, 1, 0)] == (pytest.approx(0.35), {(MAJORITY, 2): pytest.approx(0.35)})
    assert got[(3, 3, 0)] == (pytest.approx(0.35), {(MINORITY, 2): pytest.approx(0.35)})
    got = tags(m, (3, 2, 0), m.Action.Override)
    assert all(v[1] == {(MAJORITY, 2): v[0]} for v in got.values())


def test_udtb_tags():
    m = UDTieBreakingModel(0.3, 6)
    got = tags(m, (3, 2, 1), m.Action.OverrideTie)  # publish 2, win the tie
    assert set(got) == {(2, 0, 0), (1, 1, 0), (1, 1, 1)}
    assert all(v[1] == {(MAJORITY, 2): v[0]} for v in got.values())
    got = tags(m, (3, 2, 0), m.Action.OverrideMore)  # publish 3
    assert all(v[1] == {(MAJORITY, 2): v[0]} for v in got.values())
    assert all(MINORITY not in k for v in got.values() for k in v[1])


def test_shtb_tags_and_tie_probabilities():
    m = SmallestHashTieBreakingModel(0.3, 6, 5)
    # a=4, h=2 (a == h+2), last attacker block in region 1: publish 3, one block (region 1) stays
    state = (4, 2, 0, 1, 3)
    got = tags(m, state, m.Action.OverrideSafe)
    ph = 0.7 / 5
    assert got[(1, 1, 1, -1, -1)][0] == pytest.approx(ph * 3)  # honest region 2..4 loses to region 1
    assert got[(1, 1, 0, -1, -1)][0] == pytest.approx(ph * 2)
    assert all(v[1] == {(MAJORITY, 2): v[0]} for v in got.values())
