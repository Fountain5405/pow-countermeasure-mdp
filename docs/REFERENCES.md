# What the referenced research did (digest, 2026-11-17)

Scope: the works cited in Rucknium's research directions (the gist behind this
repository) and their three code bases. Five inventories in
`docs/references/` hold the details with file:line references:
`zp2019.md`, `aumayr2026.md`, `zur2020.md`, `literature.md` and `monero.md`.
Each covers the sources found (manuscript versions, appendices, code history),
the models and parameters, how each metric is computed and which bound it
gives, every published number, and where code and paper differ. Claims marked
**(verified)** were checked here by running or reading the code; the rest are
as the inventories report them.

## The common method

All three code bases follow Sapirshtein, Sompolinsky & Zohar (SSZ, FC 2016).
Each builds an MDP truncated at a maximum fork length, over the attacker's
lead, the honest lead and the fork state. The attacker's relative revenue rou
is found by bisection on (1 - rou) Rs - rou Rh with relative value iteration
(MDPtoolbox, epsilon 1e-4, stopping at epsilon/8). A forced action at the cap
is legal in the full game, so the truncated optimum is a **lower bound**.
SSZ's over-paying award at the cap gives an **upper bound**. pto (Zur et al.)
replaces the bisection with probabilistic termination (PTO). pcmdp ports the
same models and solves the ratio exactly by Howard policy iteration, so its
values are exact for the truncated MDP and are lower bounds.

## Zhang & Preneel (2019), "Lay Down the Common Metrics"

- **Scope:** 8 protocols (NC, UTB, UDTB, SHTB, PoP, RS, Fruitchains,
  Subchains) and 4 metrics: chain quality, incentive compatibility,
  subversion gain (plus a subversion bounty) and censorship susceptibility.
- **Sources:** the IEEE paper with appendices A-E, and the code at
  `nirenzang/PoWSecurity` (2019). `Results.xlsx`, added in 2021, holds the
  plotted data. No arXiv version, errata or thesis was found.
- **Bounds:** only NC, UTB and UDTB compute both bounds. SHTB, PoP, RS,
  Fruitchains and Subchains return lower bounds on attacker revenue, which is
  why the captions call them upper bounds on I and Q. SHTB uses a looser
  tolerance (8e-4).
- **Where code and paper differ:**
  - **Thresholds:** PoP's profitable threshold is 0.2640, not 0.2500, and
    SHTB's is 0.0580, not 0.0652, which is the threshold script's tolerance
    **(verified, two reviews; see monerosim's plan doc, "Independent
    reviews")**.
  - **Caps:** NC, UTB and UDTB run at lmax 80 for alpha <= 0.4 (the paper says
    160). PoP uses maxB 28 (the paper says 30), and Fruitchains IC uses lmax
    28 for To 7 (the paper says 20).
  - **Table V:** its gamma 0 and ratio 3 cases are not in the scripts.
  - **NC censorship bound:** the routine reuses the revenue cap award, so its
    "upper bound" is not valid for that objective.
  - **PoP ties:** PoP's tie share is hard-coded at 0.5.

## Aumayr et al. (2026), "Optimal Reward Allocation via Proportional Splitting"

- **Versions:** arXiv v1 (2025-03), an anonymous extended version (2026-01,
  optrevalloc.github.io), and arXiv v2 (2026-08). v2 is the AFT 2026 full
  version, renamed FairChain. The code (`commonprefix/proportional-reward-splitting-MDP`)
  dates from Dec 2024 to Jan 2025, plus a LICENSE commit in 2026-08. It was
  never updated for v2.
- **Scope:** Bitcoin, RS, PRS and Fruitchains, each for incentive
  compatibility, subversion gain and censorship susceptibility. Also window
  sweeps (R 3/6/9, k 1/3/6) and a sampling-accuracy closed form (Fig. 7, no
  code).
- **Where code and paper differ:**
  - **Bitcoin baseline:** the Bitcoin incentive-compatibility baseline is the
    capped upper-bound branch: 0.8589 stored against an optimum of 0.7596 at
    alpha 0.45, gamma 0.5 **(verified)**. The paper, code comments, commit
    messages and issues give no reason. Zhang & Preneel's original kept both
    bounds; this copy drops the lines that store them. The paper never says
    which bound or which caps (30 / 24 / 40) it uses.
  - **Introduction:** "PRS earns 38%, RS 40% at 38%" matches the R6k1 data,
    not the plotted R6k6 (PRS 38.34, RS 42.63).
  - **R = 9 claim:** "R = 9: up to 38%" does not match either: PRS stays
    honest through 39%.
  - **Fork window:** k enters only the adopt reward.
  - **Negative reward:** the RS and PRS adopt reward can be negative for
    a > h. It is not exploited at the tested points.
  - **Power estimate:** PRS splits with the true alpha, so the estimation
    error is never fed back into the MDP.
  - **Deployment parameters:** v2 gives cap 9, target 3 and freshness 2 for
    the Quai deployment, which do not match go-quai HEAD (16/32, 8 and 3).

## Zur, Eyal & Tamar (2020), "Efficient MDP Analysis for Selfish-Mining"

- **Sources:** arXiv 2007.05614 v1 and v2 (2020), and the code
  `roibarzur/pto-selfish-mining` (20 commits, last in 2023-05).
- **Models:** the paper has Bitcoin and Ethereum (uncles; threshold 0.2468).
  The repo adds a simple model, fee models and a deep-RL package (WeRLman).
  We vendored only the base classes, the Bitcoin model and the PTO and OSM
  solvers.
- **Accuracy:**
  - **Linear solver:** `pymdptoolbox-roi` 4.2.0 (2025-05) switched from exact
    sparse solves to Krylov solvers, so current runs differ from the paper's.
  - **Illegal actions:** PTO zeroes final-state transitions, which erases the
    illegal-action penalty.
  - **Stopping rule:** the stop test is absolute.
  - **Value bias:** V/H runs low by about Dmax/H.
  - **Theorem 1:** the bound is vacuous at our caps.
  - The two independent reviews found the same failures **(verified)**.
- **Oracle:** their Table 1 (Bitcoin, gamma 0, cap 95, alpha 1/3 to 0.475) is
  an oracle for `NakamotoModel`.

## Strategy and methods literature

- **SSZ:** the full gamma-0 revenue table, the over-paying upper bound, and
  the uniform tie-breaking threshold of 0.2321 (our UTB port reproduces it).
- **Eyal & Sirer:** the closed form matches SSZ's SM1 column to 5 digits.
- **Nayak et al.:** the stubborn strategies are sub-policies of SSZ's action
  set, so an exact optimum dominates them.
- **Bahrani & Weinberg (EC 2024):** undetectable selfish mining is profitable
  from alpha = (3 - sqrt 5)/2, about 0.382. The paper adds a detectability
  metric.
- **Li et al. (survey):** SquirRL and WeRLman, RL methods for state spaces too
  large to solve exactly. WeRLman's code is in the pto repo.
- **Kroll et al.:** the disruption objective's conceptual basis.
- **Carlsten et al.:** a fee model; not relevant here yet.

## Monero sources

- **MRL #146 (SoP):** the current rule dates from 2026-10-20 and the revision
  history from 2025-09: w went from 8 to 16, the uniqueness rule was added,
  uncle blocks were dropped, and tip shares started counting under
  sop_objective on 2026-10-20.
  - **The author's simulator:** alpha 1/3, stubbornness 10, gamma 0.5, and an
    80 s clock because attacker hash is left out of the difficulty. It counts
    tip shares and gives a 10+ block reorg about once per 2-3 years.
  - **An independent contributor's event simulator** reproduces those figures
    and adds three: propagation delay multiplies the 10+ rate by 4.5 at 1 s
    and by 81 at 3 s, the uniqueness rule loses 15-16% of honest shares at
    3 s, and partition recovery takes about 3.4 blocks.
- **MRL #144 (PoP for Monero):** closed on 2026-11-12 in favour of #146. It
  is a soft fork with k 3, D 5 s, one uncle at N-1 and a hash tie-break.
  - **Threshold:** the contributor's MDP gives PoP profitable from 0.264,
    which equals ours.
  - **The "on the verge" attack:** Ren Zhang advised against a faithful
    implementation because of a split-view attack that is outside his
    model and ours.
  - **Hard-fork variant:** the reward-splitting variant has never been
    simulated.
- **Rucknium (2025):** Qubic's 18-block reorg on 2025-09-14 invalidated
  transactions only from the 12th orphaned block on.
- **Lee & Kim (AFT 2026):** Qubic's hashrate share was 0.234 over the whole
  campaign, 0.283 in its active periods and 0.32-0.38 per period, with gamma
  near 0. Its revenue came out 4% below honest mining after difficulty
  effects. The realistic regime is therefore alpha 0.23-0.38 with gamma 0.
- **Research-lounge log (2025-09-01):** gamma is near 0 because blocks with
  withheld transactions relay slowly (fluffy blocks).
- **Detective Mining (Lee & Kim 2019):** not read (ePrint blocked the
  download).

## What this means for our results

1. **SoP acceptance convention.** Our SoP disruption numbers use the
   acceptance convention of monerosim's patched daemon, not #146's text. The
   daemon rejects a reveal that needs the attacker's own pending tip shares;
   the text counts them. `sop_tight.py` hard-codes the daemon's choice
   **(verified)**. Independent simulators with a weaker, fixed stubborn
   attacker report about 17 times more 10+ reorgs per block than our optimum
   at alpha 0.33. This is under investigation; results will be added here.
2. **No propagation delay.** No model here has propagation delay or SoP's
   lateness rules, and neither do the references' MDPs. The independent
   simulator shows delay dominates SoP's deep-reorg rate.
3. **Comparing with the references.** Compare with Zhang & Preneel's lower
   bounds. Do not use Aumayr's Bitcoin series above alpha 0.40. State the
   caps, gamma and units with every comparison (inventories, section 7).

## Gaps (for information)

- **Metrics:** subversion gain and bounty, and censorship susceptibility, for
  every protocol (Zhang & Preneel; Aumayr et al.).
- **Bounds:** an upper-bound (over-paying) solver in pcmdp. monerosim's
  `sop_mdp_crosscheck.py btc --bound upper` has one for Nakamoto.
- **Validation not yet done:**
  - Zhang & Preneel: the rest of Table II, Table III, most of Tables IV-V and
    the Fig. 2 curves.
  - Zur's Table 1.
  - SSZ's table beyond alpha 0.35.
  - Aumayr's subversion-gain and censorship output.json points.
- **Models:**
  - Ethereum uncles.
  - The PoP-Monero variant and the "on the verge" attack.
  - #144's hard-fork variant.
  - SoP lateness and delay, the uniqueness loss, and random ties.
  - A detectability metric.
  - RL search for state spaces too large to solve.
- **Stale:** `tests/test_pop.py`'s docstring still gives PoP's threshold as
  0.25.
