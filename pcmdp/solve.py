"""Solve a countermeasure MDP and report the policy's revenue and reorg profile.

One JSON line per run on stdout. Run on a lane box with the BLAS threads
capped (OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4); the solver needs scipy,
networkx and pymdptoolbox-roi, which the repo venv lacks.

  python -m pcmdp.solve nakamoto --alpha 0.35 --gamma 0 --max-fork 40
  python -m pcmdp.solve nakamoto --alpha 0.35 --gamma 0.5 --max-fork 60 \
      --objective disruption --min-depth 11
  python -m pcmdp.solve nakamoto --alpha 0.35 --gamma 0 --policy sm1 --mc-steps 2000000
  python -m pcmdp.solve sop --alpha 0.40 --objective disruption --exact-objects 20000000

Other protocols: rs, prs (reward splitting), utb, udtb, shtb (tie breaking), pop,
subchains, fruitchains. --max-fork defaults to each reference's value (nakamoto
and sop: 40); for pop it is the reference's maxB. --gamma is required by every
protocol that has one (nakamoto defaults to 0; udtb, shtb, pop reject it).
--normalization time makes every disruption rate per block interval (strong
block for subchains, block for fruitchains); the output "unit" says which.
Protocol-specific flags (--window, --regions, --ratio, ...) default to the
class defaults.

`sop` is the SoP walk model (sop_walk.py): disruption only, rates per block
interval; --exact-objects also runs the optimal policy through the exact SoP
game (sop_mdp.py's rules) for a rate that is legal there.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time

import numpy as np

from .fruitchains import FruitchainsModel
from .nakamoto import NakamotoModel
from .pop import PublishOrPerishModel
from .reorgs import OBJECTIVES, WEIGHTS, reorg_profile, simulate_profile
from .reward_splitting import ProportionalRewardSplittingModel, RewardSplittingModel
from .sop_walk import SoPWalkModel, exact_rate
from .subchains import SubchainsModel
from .tie_breaking import SmallestHashTieBreakingModel, UDTieBreakingModel, UniformTieBreakingModel

PROTOCOLS = {"nakamoto": NakamotoModel, "rs": RewardSplittingModel, "prs": ProportionalRewardSplittingModel,
             "utb": UniformTieBreakingModel, "udtb": UDTieBreakingModel, "shtb": SmallestHashTieBreakingModel,
             "pop": PublishOrPerishModel, "subchains": SubchainsModel, "fruitchains": FruitchainsModel,
             "sop": SoPWalkModel}
NO_GAMMA = ("udtb", "shtb", "pop", "sop")
# the reference's max fork / race length (pop: maxB), used when --max-fork is not given
DEFAULT_MAX_FORK = {"nakamoto": 40, "sop": 40, "rs": 30, "prs": 30, "utb": 80, "udtb": 160, "shtb": 40,
                    "pop": 30, "subchains": 20, "fruitchains": 30}
# protocol -> {cli attribute: constructor keyword}; given only when the flag is set
EXTRA = {"rs": {"window": "window", "fork_window": "fork_window"},
         "prs": {"window": "window", "fork_window": "fork_window", "reward_coefficient": "reward_coefficient"},
         "shtb": {"regions": "regions"}, "pop": {"k": "k", "tie": "tie"},
         "subchains": {"ratio": "ratio", "min_dw": "min_dw", "max_dw": "max_dw"},
         "fruitchains": {"window": "window", "block_to_fruit": "block_to_fruit", "max_afa": "max_afa",
                         "max_hf": "max_hf"}}


def extra_params(args: argparse.Namespace) -> dict:
    return {kw: getattr(args, attr) for attr, kw in EXTRA.get(args.protocol, {}).items()
            if getattr(args, attr) is not None}


def build_model(args: argparse.Namespace):
    p = args.protocol
    if p == "sop":
        return SoPWalkModel(args.alpha, w=args.w, min_depth=args.min_depth, x_min=args.x_min, x_max=args.x_max,
                            p_max=args.p_max, objective=args.objective)
    common = dict(objective=args.objective, min_depth=args.min_depth, weight=args.weight,
                  normalization=args.normalization)
    common.update(extra_params(args))
    cls = PROTOCOLS[p]
    if p == "nakamoto":
        return cls(args.alpha, args.gamma, args.max_fork, **common)
    if p in ("rs", "prs", "subchains", "fruitchains"):
        return cls(args.alpha, args.gamma, max_fork=args.max_fork, **common)
    if p == "utb":
        return cls(args.alpha, max_fork=args.max_fork, gamma=args.gamma, **common)
    if p in ("udtb", "shtb"):
        return cls(args.alpha, max_fork=args.max_fork, **common)
    if p == "pop":
        return cls(args.alpha, max_b=args.max_fork, **common)
    raise ValueError(f"unknown protocol {p}")


def legalize(model, policy):
    """The policy with every illegal action (one that leads to the final state) replaced by the
    first legal one, so Howard can start from a hand-built policy that is only right where it matters."""
    out = list(policy)
    for i in range(model.state_space.size):
        state = model.state_space.index_to_element(i)
        if state == model.final_state:
            continue
        for j in [out[i]] + list(range(model.action_space.size)):
            if model.final_state not in model.get_state_transitions(state, model.action_space.index_to_element(j)
                                                                    ).probabilities:
                out[i] = j
                break
    return tuple(out)


def solve(model, solver: str, horizon: int, epsilon: float, max_iter: int, policy0=None):
    from .vendor.blockchain_mdps import OSMSolver, PTOSolver

    if solver == "howard":
        from .howard import solve_average
        policy, g, iterations = solve_average(model, policy0=policy0, verbose=True)
        return policy, g, iterations
    if solver == "pto":
        s = PTOSolver(model, expected_horizon=horizon)
    elif solver == "osm":
        if model.objective != "revenue":
            raise ValueError("OSM bisects the revenue share; use PTO for disruption")
        s = OSMSolver(model)
    else:
        raise ValueError(f"unknown solver {solver}")
    # mdptoolbox prints its convergence warning on stdout, which carries the JSON
    with contextlib.redirect_stdout(sys.stderr):
        policy, value, iterations, _ = s.calc_opt_policy(epsilon=epsilon, max_iter=max_iter)
    return tuple(int(x) for x in policy), float(value), int(iterations)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("protocol", choices=sorted(PROTOCOLS))
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--gamma", type=float, default=None,
                   help="required where the protocol has one (nakamoto: default 0); udtb, shtb, pop, sop have none")
    p.add_argument("--max-fork", type=int, default=None,
                   help="fork / race length cap; default is the reference's (pop: maxB)")
    p.add_argument("--objective", choices=OBJECTIVES, default="revenue")
    p.add_argument("--min-depth", type=int, default=11,
                   help="shallowest reorg the disruption objective counts (default 11: deeper than the 10-block lock)")
    p.add_argument("--weight", choices=WEIGHTS, default="count")
    p.add_argument("--normalization", choices=("main_chain", "time", "fruits", "events"), default=None,
                   help="main_chain (default), or time: rates per block interval; fruitchains also "
                        "fruits (default) and events")
    p.add_argument("--window", type=int, help="rs, prs, fruitchains: reward window")
    p.add_argument("--fork-window", type=int, help="rs, prs: fork window")
    p.add_argument("--reward-coefficient", type=float, help="prs: default alpha")
    p.add_argument("--regions", type=int, help="shtb: hash regions")
    p.add_argument("--k", type=int, help="pop: fail-safe depth")
    p.add_argument("--tie", type=float, help="pop: honest share on the attacker's branch in a tie")
    p.add_argument("--ratio", type=int, help="subchains: weak blocks per strong block")
    p.add_argument("--min-dw", type=int, help="subchains: weight-difference floor")
    p.add_argument("--max-dw", type=int, help="subchains: weight-difference ceiling")
    p.add_argument("--block-to-fruit", type=float, help="fruitchains: block-to-fruit ratio")
    p.add_argument("--max-afa", type=int, help="fruitchains: attacker fruit cap")
    p.add_argument("--max-hf", type=int, help="fruitchains: honest fruit cap")
    p.add_argument("--w", type=int, default=16, help="sop: objects per block")
    p.add_argument("--x-min", type=int, default=-80, help="sop: walk floor")
    p.add_argument("--x-max", type=int, default=80, help="sop: walk ceiling")
    p.add_argument("--p-max", type=int, default=64, help="sop: attacker pending-share cap")
    p.add_argument("--exact-objects", type=int, default=0,
                   help="sop: simulate the optimal policy in the exact SoP game for this many objects")
    p.add_argument("--policy", choices=("optimal", "honest", "sm1"), default="optimal")
    p.add_argument("--solver", choices=("howard", "pto", "osm"), default="pto",
                   help="howard: exact average-reward policy iteration (use it when rates are tiny)")
    p.add_argument("--warm-start", help="howard: start from the policy in this .npz")
    p.add_argument("--horizon", type=int, default=10 ** 5, help="PTO expected horizon in blocks")
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--max-iter", type=int, default=200000)
    p.add_argument("--mc-steps", type=int, default=0, help="also estimate the profile by Monte Carlo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-report-depth", type=int, default=40)
    p.add_argument("--stationary", choices=("direct", "eigs"), default="direct",
                   help="how the reorg profile computes the stationary distribution")
    p.add_argument("--save-policy", help="write the policy (and the arguments) to this .npz")
    p.add_argument("--load-policy", help="skip the solve: read the policy from this .npz")
    args = p.parse_args(argv)

    if args.protocol in NO_GAMMA:
        if args.gamma is not None:
            p.error(f"{args.protocol} has no gamma")
    elif args.gamma is None:
        if args.protocol != "nakamoto":
            p.error(f"{args.protocol} needs --gamma")
        args.gamma = 0.0
    if args.max_fork is None:
        args.max_fork = DEFAULT_MAX_FORK[args.protocol]
    ok = ("main_chain", "time", "fruits", "events") if args.protocol == "fruitchains" else ("main_chain", "time")
    if args.normalization is None:
        args.normalization = "fruits" if args.protocol == "fruitchains" else "main_chain"
    if args.protocol != "sop" and args.normalization not in ok:
        p.error(f"{args.protocol}: --normalization must be one of {ok}")

    model = build_model(args)
    unit = ("block interval" if args.protocol == "sop" or args.normalization == "time"
            else {"fruits": "credited fruit", "events": "mined unit"}.get(args.normalization, "main-chain block"))
    out = {"protocol": args.protocol, "model": repr(model), "alpha": args.alpha, "gamma": args.gamma,
           "normalization": args.normalization, "params": extra_params(args),
           "max_fork": args.max_fork, "objective": args.objective, "min_depth": args.min_depth,
           "weight": args.weight, "policy": args.policy, "states": model.state_space.size,
           "unit": unit}
    t0 = time.time()
    if args.load_policy:
        policy = tuple(int(x) for x in np.load(args.load_policy)["policy"])
        if len(policy) != model.state_space.size:
            raise ValueError("the saved policy does not fit this model")
        out["loaded_policy"] = args.load_policy
    elif args.policy == "optimal":
        policy0 = (tuple(int(x) for x in np.load(args.warm_start)["policy"]) if args.warm_start else None)
        if (policy0 is None and args.solver == "howard" and args.protocol not in ("nakamoto", "sop")
                and hasattr(model, "build_honest_policy")):
            policy0 = legalize(model, model.build_honest_policy())
        with contextlib.redirect_stdout(sys.stderr):
            policy, value, iterations = solve(model, args.solver, args.horizon, args.epsilon, args.max_iter,
                                              policy0=policy0)
        out.update({"solver": args.solver, "horizon": args.horizon, "value": value,
                    "iterations": iterations})
    elif args.policy == "honest" and hasattr(model, "build_honest_policy"):
        policy = model.build_honest_policy()
    elif args.policy == "sm1" and args.protocol == "nakamoto":
        policy = model.build_sm1_policy()
    else:
        raise ValueError("this policy is not defined for this protocol")
    out["solve_s"] = round(time.time() - t0, 1)
    if args.save_policy:
        np.savez_compressed(args.save_policy, policy=np.array(policy, dtype=np.int8), args=json.dumps(vars(args)))
        out["saved_policy"] = args.save_policy

    t0 = time.time()
    out["profile"] = reorg_profile(model, policy, max_report_depth=args.max_report_depth,
                                   method=args.stationary)
    out["stationary"] = args.stationary
    out["profile_s"] = round(time.time() - t0, 1)
    if args.mc_steps:
        t0 = time.time()
        out["mc"] = simulate_profile(model, policy, args.mc_steps, seed=args.seed,
                                     max_report_depth=args.max_report_depth)
        out["mc_s"] = round(time.time() - t0, 1)
    if args.exact_objects:
        if args.protocol != "sop":
            raise ValueError("--exact-objects applies to sop")
        t0 = time.time()
        out["exact"] = exact_rate(model, policy, args.exact_objects, seed=args.seed)
        out["exact_s"] = round(time.time() - t0, 1)
    out["honest_value"] = model.get_honest_revenue()
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
