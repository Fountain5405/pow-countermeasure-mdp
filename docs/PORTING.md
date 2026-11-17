# Porting a protocol MDP into pcmdp

This is the brief for porting a reference MDP (Zhang & Preneel's PoWSecurity
or Aumayr et al.'s proportional-reward-splitting-MDP, both MATLAB, GPL-3.0)
into this package. This repository is GPL-3.0-or-later, so the MATLAB logic
may be ported directly; cite the source file and line in comments.

## Pattern

Read `pcmdp/nakamoto.py` and `pcmdp/reorgs.py` first. A protocol is a class
`<Name>Model(TaggedModel, ...)` built on pto's base
(`pcmdp/vendor/blockchain_mdps`), with:

- `get_state_space`, `get_action_space`, `get_initial_state`,
  `get_final_state` (pto's spaces: `MultiDimensionalDiscreteSpace`,
  `DiscreteSpace`, `DefaultValueSpace`), `__repr__`, `__reduce__`;
- `get_tagged_transitions(state, action, check_valid)` returning a
  `TaggedTransitions`; illegal actions go to the final state with
  `reward=self.error_penalty` (pto's convention);
- `self.init_objective(objective, min_depth, weight)` before
  `super().__init__()`, and `self.alpha` set.

**Rewards.** The reference MDPs carry two rewards per transition, `Rs`
(attacker) and `Rh` (honest), and bisect on rou in `(1-rou) Rs - rou Rh`; the
result is rou = long-run Rs / (Rs + Rh). In pto terms: `reward = Rs` and
`difficulty_contribution = Rs + Rh` on every transition. Then the ratio the
solvers compute is the reference's rou. Keep fractional rewards exactly as the
reference has them (reward splitting pays halves, proportional splitting pays
alpha / 1-alpha shares, and so on).

**Reorg tags.** Tag every transition that makes the chain most honest miners
treat as their tip lose blocks: `reorgs=[(MAJORITY, depth)]`, depth = blocks
removed from that chain. If part of the honest power (a gamma share, a
tie-break share) followed a branch that then loses, tag
`(MINORITY, depth)` on that transition. Tags do not change revenue; they feed
the disruption objective and the reorg profile. Where the reference state
does not determine the depth exactly, say so in the code and in your report
(ESCALATE: yes) instead of guessing.

**Caps.** Keep the reference's truncation (max fork / race length) as a
constructor argument with the reference's value as the default; tests use
small values.

## Files you may touch

Only `pcmdp/<your_protocol>.py` and `tests/test_<your_protocol>.py` (plus a
`scripts/check_<your_protocol>.py` for the oracle runs if you need one). Do not
edit shared files (`reorgs.py`, `howard.py`, `solve.py`, `nakamoto.py`,
`vendor/`); if you need a change there, report it. Do not commit.

## Where to run

Never run Python solves on the control box (the machine holding this
checkout): it runs only the agents. Syntax checks are fine. Run everything on
the box `box90` (ssh alias; 256 threads, 1 TB RAM, shared with other users,
disk nearly full: keep outputs small). From the checkout's root:

```
rsync -a --delete --exclude __pycache__ --exclude .git \
    ./ box90:mdp_work/port_<protocol>/
ssh box90 'cd ~/mdp_work/port_<protocol> && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    nice timeout 3000 ../venv/bin/python -m pytest -q -p no:cacheprovider -W ignore tests/test_<protocol>.py'
```

Use your own `port_<protocol>` directory (other agents run in parallel), and
write every run output and log to a separate `~/mdp_work/out_<protocol>/`:
`rsync --delete` into `port_<protocol>` deletes anything written there
(2026-11-16, an agent lost its oracle logs that way).
Cap BLAS threads at 4 and keep each job under ~64 GB (`ulimit -v 64000000`).
For runs longer than ~9 minutes, launch with `setsid nohup ... > log 2>&1 &`
and poll the log, killing by pid, never `pkill -f`.

## Solvers

- Revenue: `pcmdp.howard.solve_average(model)` returns `(policy, g, it)`;
  `g` is the exact optimum of the truncated MDP (a lower bound on the
  untruncated one). `PTOSolver` is fine for revenue too but misreports tiny
  disruption rates; prefer howard.
- Check `reorg_profile(model, policy)["revenue"] == g`.

## Done means

1. The model reproduces the oracle values given in your brief, within the
   stated tolerance, at the reference's parameters (or say exactly which
   point does not and why).
2. Tests in `tests/test_<protocol>.py`: transition probabilities sum to 1
   for every state/action; the honest policy (if the reference defines one,
   or the obvious one) earns alpha; one small oracle-style check that runs in
   under ~5 minutes on box90; reorg tags on one hand-checked transition.
3. A report: what you ported (file:line), state/action encoding, how
   rewards map, where the tags go and why, the oracle table (ours vs
   reference), runtimes, and anything you could not settle.
