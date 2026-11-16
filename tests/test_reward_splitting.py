"""Tests for the RS / PRS models (reward_splitting.py). Run on box90."""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("networkx")

from pcmdp import howard  # noqa: E402
from pcmdp.reorgs import MAJORITY, MINORITY, reorg_profile  # noqa: E402
from pcmdp.reward_splitting import (ProportionalRewardSplittingModel,  # noqa: E402
                                    RewardSplittingModel, count_blocks)

MODELS = [RewardSplittingModel, ProportionalRewardSplittingModel]


def small(cls, alpha=0.4, gamma=0.5, **kw):
    return cls(alpha, gamma, kw.pop("max_fork", 5), kw.pop("window", 3), kw.pop("fork_window", 2), **kw)


@pytest.mark.parametrize("cls", MODELS)
def test_probabilities_sum_to_one_and_tags_fit(cls):
    m = small(cls)
    for i in range(m.state_space.size):
        s = m.state_space.index_to_element(i)
        for j in range(m.action_space.size):
            t = m.get_state_transitions(s, m.action_space.index_to_element(j))
            assert math.isclose(sum(t.probabilities.values()), 1, abs_tol=1e-12)
            for nxt, tags in t.reorgs.items():
                assert sum(tags.values()) <= t.probabilities[nxt] + 1e-12


@pytest.mark.parametrize("cls", MODELS)
def test_honest_policy_earns_alpha(cls):
    m = small(cls)
    mdp, keep, P, r, d, legal, start = howard._tables(m)
    A = m.Action
    pol = np.zeros(len(keep), int)
    for j, i in enumerate(keep):
        a, h, _, _ = m.state_space.index_to_element(int(i))
        pol[j] = int(A.Override1) if a > h else int(A.Adopt)
    rows = np.arange(len(keep))
    assert legal[pol, rows].all()
    P_pi = sum(__import__("scipy.sparse").sparse.diags((pol == a).astype(float)) @ P[a]
               for a in range(len(P))).tocsr()
    g, _ = howard.evaluate(P_pi, r[pol, rows], d[pol, rows], start)
    assert g == pytest.approx(0.4, abs=1e-9)


@pytest.mark.parametrize("cls,key,expected", [
    (RewardSplittingModel, (0.35, 0.5, 3, 6), 0.39968109130859375),
    (ProportionalRewardSplittingModel, (0.35, 0.5, 3, 3), 0.36354827880859375),
])
def test_reference_oracle(cls, key, expected):
    alpha, gamma, window, fork_window = key
    m = cls(alpha, gamma, 30, window, fork_window)
    policy, g, _ = howard.solve_average(m)
    assert g == pytest.approx(expected, abs=2e-4)
    assert reorg_profile(m, policy)["revenue"] == pytest.approx(g, abs=1e-9)


def test_prs_rewards_split_by_coefficient():
    m = small(ProportionalRewardSplittingModel, alpha=0.3)
    assert m.reward_coefficient == 0.3
    # adopt at (a=3, h=1, history 0b01): shared = 1 + min(3, 2) = 3 blocks
    t = m.get_state_transitions((3, 1, m.Fork.Irrelevant, 1), m.Action.Adopt)
    nxt = (0, 1, m.Fork.Irrelevant, 0)
    assert t.rewards[nxt] == pytest.approx(3 * 0.3)
    assert t.difficulty_contributions[nxt] == pytest.approx(3 + 1 - 2)   # Rs + Rh = shared + h - min(a, fw)
    rs = small(RewardSplittingModel)
    r = rs.get_state_transitions((3, 1, rs.Fork.Irrelevant, 1), rs.Action.Adopt)
    assert list(r.rewards.values())[0] == pytest.approx(1.5)


def test_reorg_tags_hand_checked():
    m = small(RewardSplittingModel, alpha=0.4, gamma=0.5, window=3)
    F, A = m.Fork, m.Action
    # override k=1 at a=3, h=2: publishes 3 blocks, honest chain loses 2 (both successors)
    t = m.get_state_transitions((3, 2, F.Relevant, 0), A.Override1)
    assert len(t.reorgs) == 2
    assert all(tags == {(MAJORITY, 2): pytest.approx(t.probabilities[n])} for n, tags in t.reorgs.items())
    # match at a=2, h=2 (relevant): gamma branch majority depth 2, honest branch minority depth 2
    t = m.get_state_transitions((2, 2, F.Relevant, 0), A.Match)
    major = [n for n, tags in t.reorgs.items() if (MAJORITY, 2) in tags]
    minor = [n for n, tags in t.reorgs.items() if (MINORITY, 2) in tags]
    assert len(major) == 1 and len(minor) == 1
    assert t.reorgs[major[0]][(MAJORITY, 2)] == pytest.approx(0.5 * 0.6)
    assert t.reorgs[minor[0]][(MINORITY, 2)] == pytest.approx(0.5 * 0.6)
    # gamma 0: no minority tag; override with h = 0 causes no reorg
    m0 = small(RewardSplittingModel, gamma=0.0)
    t = m0.get_state_transitions((2, 2, F.Relevant, 0), A.Match)
    assert not any((MINORITY, 2) in tags for tags in t.reorgs.values())
    assert not m.get_state_transitions((2, 0, F.Irrelevant, 0), A.Override1).reorgs


def test_count_blocks():
    assert [count_blocks(x) for x in (0, 1, 6, 255)] == [0, 1, 2, 8]
