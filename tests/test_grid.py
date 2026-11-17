"""Time normalization for every protocol, the solve.py CLI, and the Adopt-from-Active tag.

Run on box90 with the BLAS threads capped (see docs/PORTING.md).
"""
import json
import math

import pytest

pytest.importorskip("scipy")
pytest.importorskip("networkx")
pytest.importorskip("mdptoolbox")

from pcmdp import solve as S  # noqa: E402
from pcmdp.fruitchains import FruitchainsModel  # noqa: E402
from pcmdp.howard import solve_average  # noqa: E402
from pcmdp.nakamoto import NakamotoModel  # noqa: E402
from pcmdp.pop import PublishOrPerishModel  # noqa: E402
from pcmdp.reorgs import MAJORITY, MINORITY, reorg_profile, simulate_profile  # noqa: E402
from pcmdp.reward_splitting import ProportionalRewardSplittingModel, RewardSplittingModel  # noqa: E402
from pcmdp.subchains import SubchainsModel  # noqa: E402
from pcmdp.tie_breaking import (SmallestHashTieBreakingModel, UDTieBreakingModel,  # noqa: E402
                                UniformTieBreakingModel)

# name -> factory(alpha, **kw)
FACTORIES = {
    "nakamoto": lambda a, **kw: NakamotoModel(a, 0.5, 6, **kw),
    "rs": lambda a, **kw: RewardSplittingModel(a, 0.5, 4, 3, 3, **kw),
    "prs": lambda a, **kw: ProportionalRewardSplittingModel(a, 0.5, 4, 3, 3, **kw),
    "utb": lambda a, **kw: UniformTieBreakingModel(a, 6, **kw),
    "udtb": lambda a, **kw: UDTieBreakingModel(a, 6, **kw),
    "shtb": lambda a, **kw: SmallestHashTieBreakingModel(a, 5, 3, **kw),
    "pop": lambda a, **kw: PublishOrPerishModel(a, max_b=6, **kw),
    "subchains": lambda a, **kw: SubchainsModel(a, 0.5, 2, 5, -3, 6, **kw),
    "fruitchains": lambda a, **kw: FruitchainsModel(a, 0, max_fork=6, window=3, max_afa=4, max_hf=6, **kw),
}
UNIT = {"subchains": 0.5, "fruitchains": 0.5}   # 1/ratio, 1/(1+b) at ratio 2, b 1; others 1


def honest_policy(model):
    """The model's own honest policy, else: override or wait while no honest block exists, adopt after."""
    if hasattr(model, "build_honest_policy"):
        return model.build_honest_policy()
    h_index = 0 if isinstance(model, PublishOrPerishModel) else 1
    names = [n for n in model.Action.__members__]
    policy = []
    for i in range(model.state_space.size):
        state = model.state_space.index_to_element(i)
        choice = model.Action.Adopt
        if state != model.final_state and state[h_index] == 0:
            for n in names:
                if n.startswith("Override") or n == "Wait":
                    t = model.get_state_transitions(state, model.Action[n])
                    if model.final_state not in t.probabilities:
                        choice = model.Action[n]
                        break
        policy.append(int(choice))
    return tuple(policy)


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_time_contribution_is_the_unit_on_every_legal_transition(name):
    m = FACTORIES[name](0.3, objective="disruption", normalization="time")
    unit = UNIT.get(name, 1.0)
    for i in range(m.state_space.size):
        state = m.state_space.index_to_element(i)
        for j in range(m.action_space.size):
            action = m.action_space.index_to_element(j)
            t = m.get_state_transitions(state, action)
            for nxt in t.probabilities:
                # Nakamoto's Adopt / Override / Match mine nothing (the next block is a later Wait)
                idle = name == "nakamoto" and action is not m.Action.Wait
                want = 0.0 if nxt == m.final_state or idle else unit
                assert t.difficulty_contributions[nxt] == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_revenue_unaffected_and_honest_has_no_disruption(name):
    plain = FACTORIES[name](0.4)
    timed = FACTORIES[name](0.4, normalization="time")
    _, g, _ = solve_average(plain)
    _, g_time, _ = solve_average(timed)
    assert abs(g - g_time) < 1e-12 and g > 0.4 - 1e-9
    dis = FACTORIES[name](0.3, objective="disruption", min_depth=1, normalization="time")
    prof = reorg_profile(dis, honest_policy(dis))
    assert prof[f"{MAJORITY}_reorgs_per_block"] == 0 and prof[f"{MINORITY}_reorgs_per_block"] == 0


def test_subchains_time_is_one_over_ratio():
    # Wait in the base state: 4 outcomes (strong / weak, honest / attacker), each one unit.
    for ratio in (2, 4):
        m = SubchainsModel(0.3, 0.5, ratio, 5, -3, 6, objective="disruption", normalization="time")
        t = m.get_state_transitions((0, 0, -1, m.Fork.Irrelevant, 0), m.Action.Wait)
        assert len(t.probabilities) == 4
        assert all(v == pytest.approx(1 / ratio) for v in t.difficulty_contributions.values())
        # a strong block is 1 unit in `ratio`: unit mean time x ratio = one strong-block interval
        assert sum(t.probabilities[n] * t.difficulty_contributions[n] for n in t.probabilities) \
            == pytest.approx(1 / ratio)


def test_fruitchains_time_is_one_over_one_plus_b():
    for b in (1.0, 3.0):
        m = FruitchainsModel(0.3, 0, max_fork=6, window=3, block_to_fruit=b, max_afa=4, max_hf=6, objective="disruption",
                             normalization="time")
        t = m.get_state_transitions((0, 0, 0, 0, 0), m.Action.Wait)
        assert len(t.probabilities) == 4
        assert all(v == pytest.approx(1 / (1 + b)) for v in t.difficulty_contributions.values())
        # P(block) = 1/(1+b): blocks take one interval, so a unit takes 1/(1+b) of it
        assert (0.7 + 0.3) / (1 + b) == pytest.approx(1 / (1 + b))


# ---- Task C: Adopt from an Active fork drops the gamma share's branch

def test_nakamoto_adopt_from_active_is_tagged():
    m = NakamotoModel(0.3, 0.5, 6)
    t = m.get_state_transitions((2, 1, m.Fork.Active), m.Action.Adopt)
    assert t.reorgs == {(0, 0, m.Fork.Irrelevant): {(MINORITY, 1): pytest.approx(1.0)}}
    assert not m.get_state_transitions((2, 1, m.Fork.Relevant), m.Action.Adopt).reorgs
    m0 = NakamotoModel(0.3, 0.0, 6)
    assert not m0.get_state_transitions((2, 1, m0.Fork.Active), m0.Action.Adopt).reorgs


def test_utb_adopt_from_active_is_tagged():
    m = UniformTieBreakingModel(0.3, 6, 0.5)
    t = m.get_state_transitions((2, 2, 1), m.Action.Adopt)
    assert t.reorgs == {(0, 1, 0): {(MINORITY, 2): pytest.approx(0.7)},
                        (1, 0, 0): {(MINORITY, 2): pytest.approx(0.3)}}
    assert not m.get_state_transitions((2, 2, 0), m.Action.Adopt).reorgs
    assert not UniformTieBreakingModel(0.3, 6, 0.0).get_state_transitions((2, 2, 1), m.Action.Adopt).reorgs


def test_rs_and_subchains_adopt_from_active_are_tagged():
    m = RewardSplittingModel(0.3, 0.5, 6, 3, 3)
    t = m.get_state_transitions((3, 2, m.Fork.Active, 0), m.Action.Adopt)
    assert len(t.reorgs) == 2 and all(v == {(MINORITY, 2): pytest.approx(t.probabilities[n])}
                                      for n, v in t.reorgs.items())
    assert not m.get_state_transitions((3, 2, m.Fork.Relevant, 0), m.Action.Adopt).reorgs
    s = SubchainsModel(0.3, 0.5, 2, 5, -3, 6)
    # a = 3 blocks, last = 0b101 -> 2 blocks private, 1 published: depth 1
    t = s.get_state_transitions((3, 2, 3, s.Fork.Active, 5), s.Action.Adopt)
    assert len(t.reorgs) == 4 and all(v == {(MINORITY, 1): pytest.approx(t.probabilities[n])}
                                      for n, v in t.reorgs.items())
    assert not s.get_state_transitions((3, 2, 3, s.Fork.Relevant, 5), s.Action.Adopt).reorgs


# ---- solve.py CLI

CLI = {
    "nakamoto": ["--gamma", "0.5", "--max-fork", "6"],
    "rs": ["--gamma", "0.5", "--max-fork", "4", "--window", "3", "--fork-window", "3"],
    "prs": ["--gamma", "0.5", "--max-fork", "4", "--window", "3", "--fork-window", "3"],
    "utb": ["--gamma", "0.5", "--max-fork", "6"],
    "udtb": ["--max-fork", "6"],
    "shtb": ["--max-fork", "5", "--regions", "3"],
    "pop": ["--max-fork", "6"],
    "subchains": ["--gamma", "0.5", "--max-fork", "5", "--min-dw", "-3", "--max-dw", "6"],
    "fruitchains": ["--gamma", "0", "--max-fork", "6", "--window", "3", "--max-afa", "4", "--max-hf", "6"],
}


@pytest.mark.parametrize("name", sorted(CLI))
def test_cli_smoke(name, capsys):
    S.main([name, "--alpha", "0.3", "--solver", "howard", "--objective", "disruption", "--min-depth", "3",
            "--normalization", "time"] + CLI[name])
    out = json.loads(capsys.readouterr().out)
    assert out["unit"] == "block interval" and out["protocol"] == name
    assert out["gamma"] == (0.5 if "0.5" in CLI[name] else 0 if "0" in CLI[name] else None)
    assert math.isfinite(out["value"]) and out["profile"]["min_depth"] == 3


def test_cli_gamma_rules_and_defaults(capsys):
    for bad in (["udtb", "--gamma", "0.5"], ["pop", "--gamma", "0.5"], ["shtb", "--gamma", "0"],
                ["rs"], ["fruitchains"], ["subchains"], ["utb"]):
        with pytest.raises(SystemExit):
            S.main(bad + ["--alpha", "0.3"])
    capsys.readouterr()
    S.main(["udtb", "--alpha", "0.3", "--max-fork", "6", "--solver", "howard"])
    out = json.loads(capsys.readouterr().out)
    assert out["gamma"] is None and out["unit"] == "main-chain block"


def test_profile_revenue_stays_a_share_under_time_rates():
    """Under normalization "time" the rates are per block interval but the revenue is still a share.

    Before 2026-11-17 the profile divided the attacker's blocks by block
    intervals for such models, so its "revenue" was attacker blocks per
    interval. Selfish mining orphans blocks, so main-chain blocks per interval
    are below 1 and the per-interval reorg rates are below the per-block ones.
    """
    kw = dict(objective="disruption", min_depth=2)
    chain = NakamotoModel(0.35, 0.0, 30, **kw)
    timed = NakamotoModel(0.35, 0.0, 30, normalization="time", **kw)
    policy = chain.build_sm1_policy()
    a, b = reorg_profile(chain, policy), reorg_profile(timed, policy)
    assert b["revenue"] == pytest.approx(a["revenue"], abs=1e-12)
    assert a["revenue"] > 0.35
    ratio = b[f"{MAJORITY}_reorgs_per_block"] / a[f"{MAJORITY}_reorgs_per_block"]
    assert 0.5 < ratio < 1.0
    deep = b[f"{MAJORITY}_deep_reorgs_per_block"] / a[f"{MAJORITY}_deep_reorgs_per_block"]
    assert deep == pytest.approx(ratio, rel=1e-9)
    mc = simulate_profile(timed, policy, 400_000, seed=3)
    assert mc["revenue"] == pytest.approx(a["revenue"], rel=0.02)
    assert mc[f"{MAJORITY}_reorgs_per_block"] == pytest.approx(b[f"{MAJORITY}_reorgs_per_block"], rel=0.05)


def _old_dense_row_pi(model, policy):
    """The pre-2026-11-17 method: one balance row replaced by a dense row of ones."""
    import numpy as np
    from scipy.sparse import csr_matrix, identity
    from scipy.sparse.csgraph import breadth_first_order
    from scipy.sparse.linalg import spsolve
    from pcmdp.reorgs import SparseBlockchainMDP
    mdp = SparseBlockchainMDP(model)
    mdp.build_mdp()
    chain = csr_matrix(mdp.get_policy_induced_chain(policy))
    reach = np.sort(breadth_first_order(chain, mdp.initial_state_index, directed=True,
                                        return_predecessors=False))
    n = len(reach)
    a = (chain[reach][:, reach].T - identity(n, format="csr")).tolil()
    a[n - 1, :] = np.ones(n)
    b = np.zeros(n)
    b[n - 1] = 1.0
    pi = np.zeros(chain.shape[0])
    pi[reach] = np.clip(spsolve(a.tocsc(), b), 0.0, None)
    return pi / pi.sum()


@pytest.mark.parametrize("name,alpha", [("nakamoto", 0.4), ("fruitchains", 0.4), ("pop", 0.35)])
def test_pinned_stationary_matches_dense_row_method(name, alpha):
    import numpy as np
    from pcmdp.reorgs import stationary_distribution
    model = FACTORIES[name](alpha)
    policy, _, _ = solve_average(model)
    _, pi = stationary_distribution(model, policy, "direct")
    np.testing.assert_allclose(pi, _old_dense_row_pi(model, policy), rtol=0, atol=1e-12)
    assert reorg_profile(model, policy)["revenue"] == pytest.approx(
        solve_average(model)[1], abs=1e-9)
