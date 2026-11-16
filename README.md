# pow-countermeasure-mdp

Markov decision process (MDP) analysis of selfish-mining and disruption
attacks against proof-of-work consensus and the countermeasures proposed for
it, with Monero in mind.

The work follows Rucknium's "Research directions for selfish mining
countermeasures in Monero"
(<https://gist.github.com/Rucknium/0b6e94811099bfab44e84562621d8fe1>,
2026-11-12):

1. put the reasonable countermeasures in one MDP framework;
2. solve each for two attacker objectives: revenue, and disruption (reorgs
   deeper than Monero's 10-block lock);
3. report the length and frequency of the attacker's reorgs directly.

## Status

| Protocol | Revenue | Disruption | Reference values reproduced |
|---|---|---|---|
| Nakamoto (longest chain) | yes | yes, with a reorg-depth histogram | Sapirshtein-Sompolinsky-Zohar 0.37077 (alpha 0.35, gamma 0) |
| Share or Perish (MRL #146) | no | yes, reorgs at or above a depth threshold | its policies run in the exact SoP game agree within 1.4 % at 11+ blocks |
| Publish or Perish, reward splitting, proportional splitting, uniform / smallest-hash / unpredictable-deterministic tie-breaking, Fruitchains, Subchains | planned | planned | Zhang & Preneel (2019), Aumayr et al. (2026) |

Results so far, with their derivations and data, are written up in the
monerosim repository: `studies/sop/docs/20261113_countermeasure_mdp_plan.md`
(<https://github.com/Fountain5405/monerosim>).

## Layout

- `pcmdp/reorgs.py`: transitions tagged with the reorgs they cause, the
  revenue / disruption objective switch, and a policy's reorg profile from its
  stationary distribution (plus a Monte Carlo estimate).
- `pcmdp/howard.py`: exact average-reward policy iteration (sparse LU). Use it
  for the disruption objective: pto's PTOSolver, through mdptoolbox-roi,
  misreports very small rates.
- `pcmdp/nakamoto.py`: Nakamoto consensus (pto's Bitcoin model with reorg tags).
- `pcmdp/sop_walk.py`, `pcmdp/sop_tight.py`: Share or Perish for the
  disruption objective. The tight model keeps both sides' pending shares and
  leaves out only SoP's 32-object window, which does not bind at 11+ blocks.
- `pcmdp/sop_rules.py`: the exact SoP game rules, for checking policies.
- `pcmdp/solve.py`: command-line driver, one JSON line per solve.
- `pcmdp/vendor/`: pto-selfish-mining's MDP base (MIT), see `VENDOR.md`.
- `scripts/`: one-off checks.

## Running

Solves need numpy, scipy, networkx and pymdptoolbox-roi
(`pip install -e .[test]`). Cap the BLAS threads, or numpy takes every core:

```
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pcmdp.solve nakamoto --alpha 0.35 --gamma 0 --max-fork 40
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pcmdp.sop_tight --alpha 0.45 --min-depth 11 \
    --x-min -40 --x-max 40 --p-max 24 --q-max 24
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest -q
```

The tight SoP model at those caps has 1.2 M states and peaks near 14 GB.

## Licence

GPL-3.0-or-later (`LICENSE`), matching the reference implementations this
work ports from: Zhang & Preneel's PoWSecurity
(<https://github.com/nirenzang/PoWSecurity>) and Aumayr et al.'s
proportional-reward-splitting-MDP
(<https://github.com/commonprefix/proportional-reward-splitting-MDP>).
`pcmdp/vendor/blockchain_mdps/` is from Zur, Eyal & Tamar's
pto-selfish-mining (<https://github.com/roibarzur/pto-selfish-mining>) under
the MIT licence (`pcmdp/vendor/LICENSE-pto-selfish-mining`).

The code started in monerosim (`studies/sop/countermeasures/`, commits
`56c7346f` to `e093a4d7`) and moved here on 2026-11-16.
