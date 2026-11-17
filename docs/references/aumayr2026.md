# Aumayr et al. (2026), "Optimal Reward Allocation via Proportional Splitting" (arXiv:2503.10185): inventory

Date of inventory 2026-11-17. Read-only; nothing run except stdlib parsing of output.json. Downloads are in `.../scratchpad/refint/src_aumayr/` (v1, v2, ext, gq, gh, optrev, web), each in its own directory.

## 1. Sources found

| Source | What | Date |
|---|---|---|
| arXiv v1 (`src_aumayr/v1/p.pdf`, sha f06b4b60...) | "Proportional Splitting (PRS)" on "PoEM2"; evaluation section 6; 14.7k words | 2025-03-13 |
| "Extended version" PDF, `optrevalloc.github.io/extended.pdf` (repo optrevalloc/optrevalloc.github.io, commit c326b99; sha e3b57f2a...) | Anonymous IEEE-format submission (names "Anonymous"), still PoEM2/PRS naming, evaluation section VI; its Fig. 7 numbers equal v2's | 2026-01-22 |
| arXiv v2 (sha 7c5f90e8..., identical to our local PDF) | Renamed FairChain; "full version of the paper to appear at AFT 2026"; 36 pp. arXiv lists only v1 and v2, no journal ref, no DOI beyond the arXiv one | 2026-08-10 |
| Code `commonprefix/proportional-reward-splitting-MDP` (cited as ref [4], "accessed 08-2026") | 45 commits by Dimitris Karakostas; one branch (main), no issues, PRs, tags or releases. All code and outputs date from 2024-12-18 to 2025-01-08 (last real commit 3fe8389, "Update Fruitchains outputs"). Only later commit: a2b845a, 2026-08-10, adds LICENSE (GPL-3.0). | |
| Fork `Rucknium/proportional-reward-splitting-MDP` | Pure mirror at 3fe8389 (2025-09-09); no changes | |
| `dominant-strategies/go-quai` (ref [5]) | HEAD 6561a9f (2026-11-13) | |

The code was therefore never updated for v2. v2's text and the data come from the Jan 2025 outputs. The repo also carries `Lay_Down_the_Common_Metrics.pdf` (Zhang & Preneel, commit 62d14ed) and a copy of MDPtoolbox. Fig. 7 (sampling accuracy) has no code in the repo (grep for chernoff/workshare/samples is empty).

### v1 -> extended -> v2 differences
- Naming and framing: v1/ext PoEM2 + PRS, "implemented on a variant of PoEM". v2: FairChain (two-level: workshare recording + PRS), no change to fork choice; adds a Qubic/Monero (Aug 2025) motivation, Sec. 3.5 (off-chain workshares), Sec. 6 Discussion and Deployment, related work (StrongChain, Subchains, Flux, uncle rewards), and a statement that PRS can sit on Nakamoto, uniform, smallest-hash, unpredictable-deterministic tie-breaking, Publish-or-Perish and PoEM.
- Theorem 1: v1/ext say R = 17 (Lemma 1 inside them says R = 16, inconsistent); v2 uses R = 16 throughout ("slightly better than FruitChains' R = 17").
- Window symbol: v1/ext use `w` for the workshare window; v2 uses `R`. `k` is the fork window in all.
- Sec. 5.4 Chernoff: v1/ext print `Pr[X <= (1+d)E[X]] < eps = exp(-d^2 n p/2)` (wrong direction); v2 prints `(1-d)`. Error bound eps: v1 0.1; ext and v2 0.05. Worked examples: v1 15% adversary, 3% inaccuracy: 6,000 shares, 469 KB; 1,000 shares: ~7%, 79 KB; 35%: 8,000, 625 KB. ext and v2: 7,832 shares, 612 KB; 1,000: 8.5%, 79 KB; 35%: 10,242, 800 KB. I checked: n = -2 ln(eps)/(d^2 p_h) with p_h = 1 - adversary power gives 7,832 and 10,242; 80 B per share. v1 Fig. 7 has a storage right-axis and x up to 8,000; v2 x up to 10,000.
- Sec. 5.2 text and Figs. 2-6 appear unchanged (same MDP data); v2 adds "Rewarding the Bad vs Punishing the Good" explanation and the headline "38% threshold" in the abstract.
- v2 adds a deployment paragraph (Sec. 6): Quai network, launched January 2025, PoEM fork choice, workshares on-chain, **hard cap 9 per block, soft target 3, ~120 B headers, freshness 2 blocks, linear staleness discount, < 1 KB per block**. go-quai HEAD does not match: `params/protocol_params.go:191-230` has MaxWorkShareCount 16 (NewMax 32), ExpectedWorksharesPerBlock 8, WorkSharesThresholdDiff 3 bits below target, WorkSharesInclusionDepth 3, P2P threshold diff 7 (max 16); `core/poem.go:179` staleness discount "factor 2 per block delay" (pre-KawPow fork only). The paper's numbers are an older (or different) configuration; no document in the repo states them.

## 2. Protocols and models (all MATLAB + Python wrapper `run.py` via matlab.engine; MDPtoolbox `mdp_relative_value_iteration`, tolerance eps/8 with eps = 1e-4)

Common: attacker alpha, honest 1-alpha, gamma = honest share mining on the attacker branch in a tie. Rewards carry two arrays Rs, Rh. IC and CS bisect on rou: (1-rou)Rs - rou Rh, rou = long-run Rs/(Rs+Rh); SG solves average reward directly.

- **Bitcoin** (`*/Bitcoin`): state (a, h, fork in {irrelevant, relevant, active}); actions adopt, override, match, wait (InitStates.m:11-13). Zhang-Preneel (ZP) / Sapirshtein-Sompolinsky-Zohar (SSZ) model. maxForkLen: IC 30, SG 24, CS 40. gamma sweeps {0, 0.5, 1} in all three. SG state adds `matchedBeforeCfn`.
- **Reward Splitting RS** (`*/RewardSplitting`): state (a, h, fork, history); history = `window-1` bits, 1 = a recent main-chain height that had an orphaned honest competitor; actions adopt, wait, match, override margin 1..window-1 (InitStates.m:10-11). Adopt pays contested heights half-half (`historySum*0.5 + min(a,forkWindow)*0.5`, InitStates.m:61-64); overrides push honest objects out of the window, so the attacker keeps those heights fully (the paper's "pushed out" clause). The fork window k enters **only** through `min(a, forkWindow)` in the adopt reward. gamma 0.5 in the paper.
- **PRS** (`*/ProportionalRewardSplitting`): identical to RS except InitStates.m:61-64: shares `rewardCoefficient` (attacker) and `1 - rewardCoefficient` (honest) replace 0.5. `run.py` sets coefficient = alpha (IC run.py:30) i.e. perfect power knowledge. In CS the code uses `alphaPower` directly.
- **Fruitchains** (`*/Fruitchains`): state (ab, hb, afA, hf, last) in IC/SG, (ab, hb, hf, last) in CS; actions adopt, override (publishes everything), wait; each step is one mined unit (honest block (1-a)/(1+b), honest fruit (1-a)b/(1+b), attacker block, attacker fruit); fruit-to-block ratio b = 1 (`bfRatio`). gamma only 0 or 1. Caps: IC maxhf 50, maxafa 40 (Init.m:16-17); SG maxafa 35, maxhf 45; CS maxhf 65, no maxafa. `window` = timeOut: attacker fruits mined while ab < window are credited immediately.

Parameter grids (run.py):
- IC: maxForkLen 30; alpha 0.25, 0.30, 0.33, 0.34...0.45 (15 values). Bitcoin gamma {0, 0.5, 1}. FC window 6, gamma {0, 1}, b = 1. RS: (k,R) in {(1,6), (3,6), (6,3), (6,6), (6,9)}, gamma 0.5. PRS: k in {3,6,1} x R in {3,6,9}, gamma 0.5.
- SG: maxForkLen 24; same alpha list; Bitcoin gamma {0,0.5,1}; RS/PRS same 5 (k,R) sets as RS IC, gamma 0.5; FC window {3,6,9}, gamma {0,1}. Result stored = average reward - alpha.
- CS: maxForkLen 40; alpha 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33...0.45 (22 values); Bitcoin gamma {0,0.5,1}; RS/PRS 5 (k,R) sets x gamma {0,0.5,1}; FC window 6, gamma {0,1}.

## 3. Metrics and how the code computes them

- **Incentive compatibility (IC)**: optimal long-run attacker share of rewards, rou. Fair = alpha. Plotted as "Adversarial rewards (%)". RS, PRS, FC: lower-bound bisection only (`SolveMDP.m:5-20`, eps 1e-4, bracket [0,1], stop at width eps/8, RVI eps/8); the upper-bound block in RS/PRS is commented out (RS SolveMDP.m:25-65). Illegal actions pay Rh = 10000 into a base state. No truncation award, so these are lower bounds of the untruncated game (maxForkLen 30).
  **Bitcoin IC is different** (`Bitcoin/SolveMDP.m`): lines 10-21 lower bisection; line 23-25 `lowerBoundRou = rou; lowRou = rou; highRou = min(rou + 0.1, 1)`; lines 26-55 second bisection with the over-paying truncation reward at a == maxForkLen (:31-35) and h == maxForkLen (:37-43) from ZP; `Init.m:1,22` returns `lowRou`, i.e. the upper-bound bracket, capped at lower + 0.1. Line 49 `rouPrime` is unused. Our earlier finding stands (0.8589 stored vs 0.7596 optimum, alpha 0.45, gamma 0.5; the independent review also showed the cap is hit for all gamma at alpha 0.45; at 0.40 stored 0.5754 vs lower 0.5725).
- **Subversion gain (SG)**: double-spend reward DSR = 3 per orphaned block beyond confirmation count cfN = 6 (Init.m:13-14): `DSRcur = DSR*(h-cfN+1)` when h >= cfN and the attacker has not matched before cfN (state bit `matchedBeforeCfn`) (RS InitStates.m:47-51; Bitcoin :49-51). A failed attack pays nothing and costs nothing. Objective: average reward per step (`SolveMDP.m:7`, Bitcoin) minus alpha, printed as "Adversarial rewards (%)". FC multiplies by (1 + b) (SolveMDP.m:9) to convert per unit to per block. Truncated lower bound only (maxForkLen 24). `SubversionBounty.m` (alpha 0.1-0.4, gamma 0.95, cfN 3 and 6, maxForkLen 80/160, bisection on DSR) is the unused ZP leftover.
- **Censorship susceptibility (CS)**: same MDPs with attacker reward = honest rewards destroyed: Bitcoin override Rs = h instead of h+1 (CS InitStates.m:60-61); RS/PRS override Rs = settledBlocks (no `+margin`) (diff vs IC at lines 95-103); FC no afA state. Lower-bound bisection, maxForkLen 40.
- **Window sweeps**: Fig. 5 R in {3,6,9} at k = 6; Fig. 6 k in {1,3,6} at R = 6; IC only ("all metrics are affected in the same manner"; the data for SG and CS exist). The paper gives the reason for k: lower k penalises forking more.
- **Sampling accuracy** (Sec. 5.4, Fig. 7): closed form only, see Sec. 1.
- **Threshold scripts**: `incentive_compatibility/{RewardSplitting,ProportionalRewardSplitting}/Threshold.m` bisect alpha for the first rou > alpha + 1e-4 (alpha range 0.3702-0.3787, gamma 0, timeOut 9, maxForkLen 30). They never set `forkWindow`, so they predate the k parameter and would not run as is (commit d726292). `Init2.m` dumps policy and probability files (alpha {0.35,0.4,0.45}, R 2..9, k 2..10, gamma 0.5) for `print_strategy.py` (R 5, k 3, alpha 0.3 hard-coded); the `strategies/` outputs are not committed.
- **Figure scripts**: root `plot.py` -> `incentive_compatibility.pdf`, `subversion_gain.pdf`, `censorship_susceptibility.pdf` (Figs. 2, 3, 4; R = k = 6; RS/PRS/Bitcoin only gamma 0.5; FC all gammas at window 6; y = 100*value). `incentive_compatibility/plot.py` -> `ic_comparison_k.pdf` (Fig. 5) and `ic_comparison_w.pdf` (Fig. 6); `subversion_gain/plot.py` -> `rs_prs_comparison_{k,w}.pdf`; `censorship_susceptibility/plot.py` -> `parameter_{k,w}.pdf` (SG/CS versions of Figs. 5-6, not in the paper).

## 4. Published numbers (percent; from output.json; alpha columns 25 30 33 35 37 38 40 42 45; SG = 100*(avg reward - alpha))

IC (Figs. 2, 5, 6; gamma 0.5 unless noted):
- Bitcoin g0.5: 25.00 32.69 38.47 43.03 48.26 51.14 57.54 65.82 85.89 (g0: 25.00 30.00 33.06 37.09 41.49 43.87 49.09 56.19 75.87; g1: 33.33 42.86 49.26 53.86 58.77 61.35 66.80 72.65 91.77)
- FC g0: 26.90 34.66 40.33 44.64 49.42 51.98 57.42 63.19 72.09; FC g1: 27.19 35.47 41.66 46.40 51.65 54.46 60.35 66.45 75.43
- RS R6k6: 25.00 30.00 33.00 35.00 39.77 42.63 49.08 56.60 70.11; PRS R6k6: 25.00 30.00 33.00 35.00 37.00 38.34 45.69 54.17 69.01
- RS R3k6: 25.00 30.00 35.43 39.97 45.24 48.14 54.50 61.67 74.04; PRS R3k6: 25.00 30.00 33.00 36.48 41.99 45.17 52.18 60.02 73.30
- RS R9k6: 25.00 30.00 33.00 35.00 37.00 39.33 45.61 53.17 67.27; PRS R9k6: ... 37.00 38.00 41.56 50.22 65.91
- RS R6k1: ... 37.42 40.20 46.61 54.27 68.32; PRS R6k1: ... 37.00 38.00 43.56 52.08 67.32
- RS R6k3: ... 39.19 41.99 48.38 55.90 69.54; PRS R6k3: ... 37.00 38.00 45.08 53.54 68.47
- First alpha with value > alpha + 0.1 point: RS R6k6 0.36, PRS R6k6 0.38; RS R9 0.38, PRS R9 0.40; PRS R3 0.35, RS R3 0.33; PRS k1/k3 0.39; Bitcoin g0.5 0.30, g0 0.34, g1 0.25; FC 0.25.

SG (Fig. 3; gamma 0.5; FC window 6):
- Bitcoin: 0 0 1.4 5.3 11.3 15.2 24.5 35.7 54.1 (g0: 0 0 0 0 5.1 9.3 19.0 30.1 48.7; g1: 2.6 7.5 12.8 17.7 24.0 27.7 36.0 45.7 61.3)
- FC g0: 2.3 6.5 11.1 15.2 20.2 23.1 29.3 36.0 46.2; FC g1: 5.0 11.9 18.4 23.8 29.9 33.2 39.9 46.6 55.9 (windows 3, 9 and gamma-0/1 versions in output.json)
- RS R6k6: 0 0 2.1 7.3 14.1 18.1 27.1 37.5 54.7; PRS R6k6: 0 0 0 3.8 11.1 15.4 25.1 36.1 54.0
- PRS/RS other cells: R3k6 PRS 0 0 0.5 6.6 14.0 18.2 27.8 38.7 56.1; R9k6 PRS 0 0 0 1.5 8.7 12.9 22.5 33.7 52.0; R6k1 PRS 0 0 0 0.3 7.5 11.8 21.7 33.2 51.8; R6k3 PRS 0 0 0 2.4 9.6 13.9 23.7 34.8 53.0 (RS in the file).
- PRS exceeds Bitcoin g0.5 by 0.2-0.6 points at alpha 0.38-0.42 (15.4 vs 15.2; 25.1 vs 24.5; 36.1 vs 35.7); at 0.45 they are 54.0 vs 54.1.

CS (Fig. 4; alpha columns 10 15 20 25 30 35 40 45):
- Bitcoin g0.5: 6.2 10.4 15.8 22.6 31.3 42.5 57.1 75.9 (g0: 1.9 4.2 7.7 12.3 18.7 28.7 42.7 63.6; g1: 11.1 17.6 25.0 33.3 42.9 53.8 66.7 81.8)
- FC window 6 g0: 0.1 1.0 3.9 10.0 20.1 34.5 52.3 70.6; g1: 0.1 1.1 4.4 11.3 22.9 39.0 57.9 75.7. (output.json also holds window 7: g0 0.0 0.5 2.6 7.7 17.2 31.6 50.0 69.3.)
- RS R6k6: 5.6 8.9 12.8 17.8 24.6 34.5 49.0 70.2; PRS R6k6: 1.1 2.8 5.5 10.1 17.4 28.8 45.6 69.1
- R/k variants and gamma 0, 1 for RS/PRS are in output.json (e.g. PRS R9k6: 1.1 2.7 5.1 8.9 15.1 25.2 41.4 66.0).

Sampling accuracy: values in Sec. 1. Deployment numbers: Sec. 1.

Counts in output.json: IC 45/75/134/30 (Bitcoin/RS/PRS/FC), SG 45/105/105/90, CS 66/308/308/88.

## 5. Code vs paper: discrepancies and undocumented choices

1. Bitcoin IC baseline is the capped upper-bound branch (above). No commit message or comment explains it; the commits are terse ("Add Bitcoin IC output", 48aad39; code b3d14ea). ZP's original `Incentive Compatibility/Bitcoin/SolveMDP.m` keeps `lowerBoundReward` and `upperBoundReward` separate and printed (`disp`); Aumayr's copy removed the `disp`/store lines and `Init.m` returns the global `lowRou` (Init.m:1,22), so the upper-bound bracket leaks out. The `min(rou+0.1,1)` cap (line 25) and `maxForkLen` are ZP's (ZP used 20; this repo 30). RS/PRS/FC in ZP and here return the lower bound (Init.m returns `rou`).
2. The paper never says which bound is plotted. It says only that the evaluation yields "a lower bound on the protocol's performance" from the adversary's viewpoint; v2 adds "reported values correspond to expected long-run rewards induced by the optimal policy" for each finite MDP. Truncation caps (30/24/40) are not stated in the paper.
3. Stray entries in output.json not produced by current run.py: IC PRS gamma 0 (9) and gamma 1 (5) points; PRS IC lacks the (R9,k1) cell that run.py would produce; CS FC window 7 (run.py loops window 6 only); CS RS and PRS gamma 1 lack (R9,k6). Bitcoin IC/SG/CS gamma 0 and 1 are not in the paper (only gamma 0.5 plotted).
4. Intro claim "at 38% power PRS earns 38%, RS 40%": Fig. 2 data at R6k6 give PRS 38.34, RS 42.63. RS reaches 39.77 at 37%. The pair (38.00, 40.20) is R6k1; (38.00, 39.33) is R9k6. The "deviation threshold 38%" is right for R6k6 (PRS honest through 37%, first excess at 38%).
5. Sec. 5.3 "R = 9: PRS optimal up to 38%": data show PRS R9k6 honest (within 0.1 point) through 39%, first excess at 40%; RS R9 honest through 37%. "R = 3: up to 33%": PRS at 34% is 34.10 (borderline).
6. SG text "PRS best between 30% and 38%": at 38% Bitcoin (15.2) is below PRS (15.4). Fig. 3's y-label "Adversarial rewards (%)" is really gain over fair share.
7. RS/PRS adopt reward is not clipped for a > h: Rs = 0.5 min(a,k) + ..., Rh = ... + h - min(a,k) can be negative (InitStates.m:61-64). Adopting with a longer private chain is irrational, and results at alpha 0.25 equal alpha to 1e-5, so the optimiser does not exploit it at the tested points, but it has not been checked at high alpha.
8. k enters only the adopt reward (InitStates.m:61-64), not override or wait. The paper describes k as an eligibility rule for objects on forks; the MDP implements an approximation.
9. Paper says FC gamma 0.5 impossible "due to complexity issues" (fn. 11); the code restricts gamma to {0,1} by construction (Init.m:6-7). Paper says FC fruit window 6, ratio 1 ("following ZP baseline"); data agree.
10. PRS's proportional split uses the true alpha (rewardCoefficient = alpha). The paper states this (fn. 10) and defers estimation error to Sec. 5.4; the estimation error is never put back into the MDP.
11. Threshold.m and `SubversionBounty.m` are stale (see Sec. 3). `run.py` loops are partly commented (`# for gamma in ...`), so outputs reflect an earlier run configuration.

## 6. Coverage in pcmdp (`/home/selfishmining/pow-countermeasure-mdp`)

| Protocol | IC (revenue share) | SG | CS | Reorg / disruption extension |
|---|---|---|---|---|
| RS | yes, 75/75 output.json points within 7.8e-6 (`validation/aumayr_table_rs.txt`) | no | no | yes (tags, time normalisation) |
| PRS | yes, 134/134 within 1.7e-5 | no | no | yes |
| Fruitchains | yes, 30/30 within 4.1e-5 (max_afa 40, max_hf 50, gamma 0/1) | no | no | yes |
| Bitcoin / Nakamoto | lower bound yes (max_fork 30 matches Octave lower bound within 7e-6, blind review `q1_mf30.log`); **the reference's capped upper-bound baseline is not reproduced** in a pcmdp solver (only by Octave runs in `validation/reviews_20261117`); `solve.py` has no upper-bound mode | no | no | yes |

Gaps, concretely:
1. Subversion gain for all four: needs DSR 3, cfN 6, the `matchedBeforeCfn` bit, maxForkLen 24, average reward per step objective, gain = r - alpha; FC per-unit to per-block (x(1+b)), maxafa 35, maxhf 45, windows 3/6/9.
2. Censorship susceptibility for all four: the Rs changes in Sec. 3 (Bitcoin Rs = h; RS/PRS override Rs = settledBlocks; PRS adopt coefficient; FC without afA, maxhf 65), maxForkLen 40, alpha 0.10-0.45 (22 values), gamma 0/0.5/1.
3. Window sweeps: IC grids for R in {3,6,9}, k in {1,3,6} are covered through the RS/PRS grids, but SG and CS sweeps and the cells run.py would create but output.json lacks ((R9,k1) IC PRS; CS gamma 1 (R9,k6)) are not run.
4. Over-paying upper-bound Nakamoto/Bitcoin solver with the ZP truncation reward and `min(rou+0.1,1)` cap: absent, so the paper's Fig. 2 Bitcoin series cannot be regenerated by pcmdp (only the correct lower bound).
5. Sampling-accuracy closed form (Fig. 7) and any estimation-noise MDP: not present; PRS uses coefficient = alpha.
6. Threshold search (Threshold.m) and policy dumps (Init2.m, print_strategy.py): not present; pcmdp has `--save-policy`.
7. FC gamma 0.5 is impossible in the reference model and in pcmdp; the paper plots gamma 0 and 1 only.
8. No fork-window rule beyond the adopt reward (same as the reference), so any "true k eligibility" model is a new design, not a port.

## 7. What should change in how we run or compare

1. Never compare against the stored Bitcoin IC series above alpha 0.40. Use the optimum (0.7596 at alpha 0.45, gamma 0.5, max_fork 80; 0.7589 at 30) and say the paper's baseline is a capped upper bound. At alpha 0.44-0.45 stored = lower + 0.1.
2. RS/PRS/FC are truncated lower bounds at 30 (IC), 24 (SG), 40 (CS); report caps with every number and compare protocols at equal caps. The relevant error is 1e-3 or less at caps of 30 for these protocols, but SG at 24 near alpha 0.45 has not been checked.
3. Use the paper's exact parameter set when reproducing figures: R = k = 6; gamma 0.5 (Bitcoin, RS, PRS), gamma 0 and 1 (FC); b = 1; DSR 3, cfN 6; coefficient = alpha. Keep metric definitions distinct: IC and CS are shares (rou, ratio-bisected), SG is average reward per step (per block for FC after x(1+b)) minus alpha; our revenue/disruption objectives use per-block normalisation, so convert before comparing.
4. Define "threshold" as in the paper's text: largest alpha with value <= alpha + 0.1 point (our computation: RS 35%, PRS 37% honest, so the "38% deviation threshold" means first excess at 38%). Use R6k6, not the intro's mix.
5. When we add SG and CS, validate against the 105 + 105 + 90 + 45 SG and 308 + 308 + 88 + 66 CS output.json points exactly as done for IC (tolerance about 5e-5).
6. Add a high-alpha check that no RS/PRS policy chooses Adopt with a > h (finding 7).
7. Treat the deployment numbers (cap 9, target 3, freshness 2) as a claim without code in either repo; go-quai HEAD uses 16/32, 8 per block, depth 3.
8. Fruitchains costs 3000-8300 s per point; plan RS/PRS grids (30 s each) first.
