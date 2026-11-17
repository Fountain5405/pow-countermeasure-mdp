# Literature inventory for pcmdp (2026-11-17)

Working copies (untrusted text, converted with stdlib scripts; no pdftotext on this box):
`<scratch>/refint/src_lit/`
(`h_1507.06183/p.txt` SSZ, `h_1311.0243/p.txt` Eyal-Sirer, `nayak/c.txt` Nayak (OCR-ish, ligatures and Greek letters lost), `h_2309.06847/p.txt` Bahrani-Weinberg, `h_2502.17307/p.txt` survey, `kroll/c.txt`, `carlsten/c.txt` + `blog.txt`).
Marking: **[read]** = taken from the fetched text; **[derived]** = I computed it from a read formula; **[memory]** = not checked against a source here.
What pcmdp has: from README, PORTING.md, module docstrings, `tests/` greps at repo HEAD 4d9c66f.

## 1. Sapirshtein, Sompolinsky, Zohar (SSZ), FC 2016
Sources: arXiv 1507.06183 (full, with appendices; HTML fetched). Springer doi 10.1007/978-3-662-54970-4_30. **No author code found** (GitHub API search: nothing; the paper used MATLAB's MDPtoolbox relative value iteration of Chades et al. and a Java simulator). The maintained reimplementation is Zur-Eyal-Tamar's pto-selfish-mining (MIT, vendored in pcmdp).

Model [read]: state (a, h, fork), fork in {irrelevant, relevant, active}; actions adopt / override (a>h) / match (a>=h, needs relevant fork) / wait; one transition per mined block; gamma = share of honest power that mines on the attacker's branch in a tie; reward = pair (attacker blocks, honest blocks) accepted; objective REV = liminf sum r1 / sum(r1+r2) (a ratio, not an MDP). Honest policy gives REV = alpha. Also: 5.4 uniform tie-breaking (match success fixed 1/2), 6 a delay model (threshold goes to 0), 7 double-spend.

Method [read]:
- Binary search on rho; MDP M_rho with reward w_rho = (1-rho)x - rho y; the optimum is the rho with value 0 (Prop. 2). Algorithm 1 halts when high-low < eps/8, solving to eps/8.
- Truncation at max(a,h)=T. Under-paying MDP M^T: only adopt allowed at T (lower bound on REV). Over-paying MDP N^T: at max(a,h)=T the attacker is paid
  - T=a>=h: (1-rho)*alpha(1-alpha)/(1-2alpha)^2 + (1/2)((a-h)/(1-2alpha) + a + h);
  - T=h>=a: (1-(alpha/(1-alpha))^(h-a))*(-rho*h) + (alpha/(1-alpha))^(h-a)*(1-rho)*(alpha(1-alpha)/(1-2alpha)^2 + (h-a)/(1-2alpha)).
- Upper bound = rho' + 2(u + eps'), rho' = max(low - eps/4, 0), u the value of N_rho'^T (Prop. 5). Parameters: figures eps=1e-5, T=75; Table 2 T=95.
- Trivial bound alpha/(1-alpha), tight at gamma=1 (Prop. 1).
- Profit threshold (Cor. 1): solve N^T with "honest" removed (no override at (1,0), no adopt at (0,1)); value <= -eps proves honest optimal at that alpha.
- Validation: 1e7-round Java simulation matched the solver to +-1e-6.

**Table 2 (gamma=0), complete [read]** (alpha: SM1 / eps-OPT / upper bound):

| alpha | SM1 | eps-OPT | Upper |
|---|---|---|---|
| 1/3 | 1/3 | 0.33705 | 0.33707 |
| 0.35 | 0.36650 | 0.37077 | 0.37079 |
| 0.375 | 0.42118 | 0.42600 | 0.42604 |
| 0.4 | 0.48372 | 0.48866 | 0.48904 |
| 0.425 | 0.55801 | 0.56808 | 0.57226 |
| 0.45 | 0.65177 | 0.66891 | 0.70109 |
| 0.475 | 0.78254 | 0.80172 | 0.90476 |

Other numbers [read]: uniform tie-breaking, gamma=0.5, profit threshold 0.25 -> 0.2321 (attackers with 0.2321<alpha<0.5 gain). 32.9% is quoted for the optimal gamma=0 threshold by Bahrani-Weinberg. Fig. 2 thresholds for other gamma, Fig. 1 curves (gamma=0.5, 1) and Tables 3 (alpha=0.45, gamma=0.5, a,h<=8) and 4 (alpha=0.35, gamma=0) optimal policy grids exist; Tables 3-5 are in `h_1507.06183/p.txt` lines 346-570 (mangled table text, usable).

Relevance to pcmdp:
- Covered: the Nakamoto model (pto BitcoinModel is the SSZ model), exact policy iteration, one row (0.35: 0.370754 vs 0.37077), tie-break threshold 0.2321.
- Missing: (a) the over-paying upper bound for Nakamoto. I grepped `pcmdp/` and `solve.py` for bound/upper/overpay and found nothing, though the monerosim SoP code has one. Its formula is above; it gives SSZ's 0.37079/0.42604/... as direct tests. (b) the other six Table 2 rows (only 0.35 is tested). The alpha=0.45/0.475 rows are a good cap-convergence test: eps-OPT 0.66891 vs upper 0.70109 at T=95. (c) the honest-forbidden threshold test (Cor. 1) as an independent threshold method next to `bisect_pt`. (d) policy-grid comparison against Tables 3/4 and SM1 value 0.36650.
- Note that SSZ's relative revenue includes the whole accepted history, so the optimum and the Aumayr "capped upper bound" are the same kind of quantity as the 0.8589-vs-0.7596 finding in the latest commit: the over-paying award is what makes a capped value a true upper bound.

## 2. Eyal & Sirer (2014), "Majority is not enough"
Sources: arXiv 1311.0243 (6 pp.; HTML fetched). Also FC 2014, CACM 2018. Code: none (their simulator is not released; "Monte Carlo simulator, 1000 miners").

Model [read]: pool of size alpha, honest majority, gamma = honest fraction mining on the pool's block in a 0' tie; strategy Selfish-Mine (lead states 0, 0', 1, 2, 3+; reveal on lead 2 when honest closes to 1; publish at lead 1 on honest block). Closed form:
R_pool = [alpha(1-alpha)^2 (4alpha + gamma(1-2alpha)) - alpha^3] / [1 - alpha(1 + (2-alpha)alpha)].
Threshold (1-gamma)/(3-2gamma) < alpha < 1/2; gamma=0 -> 1/3, gamma=1/2 -> 1/4, gamma=1 -> 0. Proposed fix: propagate all equal-length branches and pick uniformly (gamma=1/2, threshold 1/4).

Validation numbers **[derived]** from the formula (I evaluated it by hand-run Python on the control box; it is closed form, not a solve):
- gamma=0: alpha 0.35, 0.375, 0.4, 0.425, 0.45, 0.475 -> 0.36651, 0.42118, 0.48372, 0.55801, 0.65177, 0.78255. These match SSZ's SM1 column to 5 digits (SSZ prints 0.36650 for the first and 0.78254 for the last), so SM1 as a fixed policy on pcmdp's Nakamoto model should reproduce them.
- gamma=0.5: R(0.25)=0.25 exactly, R(0.4)=0.5256, R(0.45)=0.6806. gamma=1: R(0.4)=0.5674. gamma=0.9: R(0.4)=0.5591.

Relevance: the SM1 policy is already runnable (`solve.py --policy sm1`), so the closed form is a free exact check across all (alpha, gamma) for the revenue objective and for the first-seen tie rule; it is not yet in tests as far as the greps show. It also pins down what "selfish mining with gamma" means for the Share-or-Perish/tie-breaking variants (gamma=1/2 equals uniform tie-breaking only for the attacker's match; SSZ shows the optimum under it is lower-threshold, 0.2321).

## 3. Nayak, Kumar, Miller, Shi (2016), "Stubborn Mining"
Sources: ePrint 2015/796 (eprint.iacr.org blocks plain curl with a Cloudflare challenge; I got the PDF from the Wayback Machine: `web.archive.org/web/2023/https://eprint.iacr.org/2015/796.pdf`). IEEE EuroS&P 2016. Code: none by the authors; third-party simulators only (diba-m/StubbornMining, MohammedGuniem/Selfish-GHOST-and-Stubborn-Mining; no licence, simulations, not worth porting).

Model [read]: Eyal-Sirer chain with parameters alpha (attacker), beta (eclipsed victim share Lucy), gamma. Family of strategies as modifications to selfish mining:
- L (lead-stubborn): at lead>=2 and honest block arrives, reveal only one block (match) instead of the whole chain; 
- F (equal-fork-stubborn): after winning an equal-length fork, do not reveal the new block;
- T_j (trail-stubborn): keep mining on the private chain until j+1 behind. Combinations LF, LT1, FT1, LFT1, plus T2 variants. Paper finds T2 never beat T1 in checked regions.
Eclipse extensions (Collude / Destroy) exist; not relevant to pcmdp.

Numbers [read]: L-stubborn up to 13.94% more gain than selfish mining at typical parameters; at alpha=0.4, gamma=0.9 the best stubborn strategy "outperforms selfish mining by 23% and honest mining by 63%"; trail-stubbornness helps by up to 13% when alpha>0.33; selfish mining is the dominant strategy only in a narrow region (Fig. 7b). Gains in the paper are normalised by alpha: **[derived]** with SM1 0.5591 at (0.4, 0.9), (R_best - 0.5591)/0.4 = 0.23 gives R_best ~ 0.652 (63% over 0.4 = 0.652). So the Nakamoto optimum at alpha=0.4, gamma=0.9 must be >= ~0.65 (rounded inputs; treat as a +-0.005 check; my reading of "gain" is an inference).

Relevance: all three families are strict sub-policies of SSZ's action set: L is `match` at a>h on an honest block (SSZ allows match for a>=h), F is `wait` instead of `override` at a>h in an active fork, T_j is `wait` instead of `adopt` while h-a<=j. So no new model is needed; pcmdp can validate stubborn policies as fixed policies and show the SSZ optimum dominates every one. Missing: these policies as named presets. They matter for the disruption objective: stubborn strategies are the ones that keep long private branches, so deep reorgs come from L/F/T_j, not SM1 (the monerosim SoP campaign calls its strategies "stubborn"; this is the same Nayak family).

## 4. Bahrani & Weinberg (2024), "Undetectable Selfish Mining"
Sources: arXiv 2309.06847 (v1 Sep 2023, v2 04 Feb 2024; v2 HTML fetched), ACM EC 2024 doi 10.1145/3670865.3673485. **Code: none found** (GitHub searches returned nothing; the paper is pure theory plus exact Markov chains, Appendix C).

Model [read]: Nakamoto Consensus Game, discrete steps, one miner per step; two players after the reduction to the strategic miner (alpha) and honest. Honest miners use longest chain with a deterministic tie-break, here *against* the attacker (no gamma; "strong selfish mining" = ties in the attacker's favour, gamma=1, revenue alpha/(1-alpha)). Heights are labelled Single or Pair. Undetectability: the view (all broadcast blocks) is identical in distribution to an honest world with latency l', i.e. each height is a Pair independently with probability beta = alpha l'(1-alpha) l' / (1 - (1-alpha)l' alpha l'). Strategy: selfish-mining skeleton that deliberately "wastes" a hidden lead (publishes without orphaning) so the Pair pattern stays i.i.d. beta; each of Attacker's first blocks is labelled Pair with probability beta/P_h where P_h = P(attacker block at height h is safe).

Key numbers [read]: Theorem 3: for tie-break in the attacker's favour, a profitable undetectable strategy exists for every alpha. Theorem 4/5: tie-break against the attacker, profitable and undetectable for alpha >= (3-sqrt5)/2 = 0.381966 (alpha >= 0.382 with latency l>0, all beta' in [0,1]); the bound is "not tight for our strategy" and the exact Markov chain (Appendix C) would give it; they cite SSZ's 32.9% as the tight threshold of detectable selfish mining. Selfish mining: Eyal-Sirer 1/3.

Relevance: the undetectability constraint is a distributional constraint on the orphan pattern, i.e. a constrained MDP (Lagrangian or fixed-beta labelling), and it is the opposite end of the disruption objective: a strategy that must look like honest latency cannot make many consecutive Pairs, so (my inference, not stated in the paper) its reorg depth distribution is geometric-like in beta, far from 11+. Concrete use: (a) the 0.382 threshold is a check value for a Nakamoto first-seen model with constrained reorg frequency; (b) a "detectability" metric (orphan rate per height vs honest-latency baseline) can be read from the same stationary distribution `reorgs.py` already computes, as a third output next to revenue and disruption. Not covered by pcmdp; low priority unless the owner wants detection-based countermeasures (a countermeasure could target exactly the orphan-pattern test they describe).

## 5. Li et al. (2025), survey "Strategic Mining ... a Reinforcement Learning Approach"
Sources: arXiv 2502.17307 (v2 25 Feb 2025; HTML fetched). No code (survey).

Content [read]: Section 2 recalls the SSZ MDP (states (la, lh), actions adopt/override/match/wait). Section 3 covers RL: Q-learning, MC, TD, policy gradient; SquirRL (Hou et al. 2019; deep RL, multi-agent; rediscovers the optimal Bitcoin policy, finds a Casper FFG attack), WeRLman (Bar-Zur et al. 2022, MCTS + DQN with variance reduction; handles transaction-fee "whales"), Bar-Zur 2023 (partially compliant small miners), Sarenche 2024 (DQN, longest-chain PoS). Table 2 thresholds quoted [read]: Hou 0.25 (Bitcoin); Bar-Zur 2022: 0.20/0.17/0.12 after 3/4/5 halvings of the minting rate; Bar-Zur 2023: 0.21 and 0.19 (0.5 / 0.75 petty-compliant miners; 25% -> 21% with half the miners compliant); Sarenche 0.24198 (PoS); SSZ 23.2% (their number). Section 4.2 ("tricks such as imposing artificial limits") says truncation is the standard way to keep state spaces finite; the survey names scalability of MDPs as the case for RL but offers no benchmark of exact vs RL accuracy.
Methods that scale (not in the survey text, [memory]): Zur-Eyal-Tamar PTO (arXiv 2007.05614, abstract fetched: solves the ratio MDP as one standard MDP with random termination; error inversely proportional to the expected number of rounds; order of magnitude cheaper than SSZ's bisection; tightens thresholds for Ethereum-like and fee models); WeRLman code lives in the same repo (`reinforcement_learning/` with DQN, MCTS-difference DQN, lookahead DQN) at github.com/roibarzur/pto-selfish-mining (MIT; default branch `master`; models: bitcoin, bitcoin fee variants, ethereum, simple).

Relevance: pcmdp already vendors PTO's MDP base and uses exact policy iteration. For state spaces that do not fit (SoP tight model at 1.2 M states, 14 GB), the survey's useful pointers are: (1) PTO (already there) as a one-solve alternative to bisection, (2) WeRLman/MCTS-DQN from the pto repo as a lower-bound search on a sampled environment (policy evaluated exactly by `reorgs`/`howard` afterwards), (3) SquirRL for the multi-attacker case (not needed). The survey's cap-as-trick statement matches our "capped solves are lower bounds" rule. Nothing in it is a model we lack.

## 6. Kroll, Davey, Felten (2013), "The Economics of Bitcoin Mining, or Bitcoin in the Presence of Adversaries"
Source: WEIS 2013, PDF via Wayback (`web.archive.org/web/2022/https://www.econinfosec.org/archive/weis2013/papers/KrollDaveyFeltenWEIS2013.pdf`; the weis2013 mirrors returned 500/blank). Code: none.
Model [read]: mining game with monotonic strategies; any monotonic strategy S played by everyone is a Nash equilibrium, so there are infinitely many equilibria besides the reference client (Section 3). Section 4: attacks that destabilize consensus on rules or state: a cartel (Goldfinger attacker, outside-the-system motive) can fork or double-spend; 51% attack. Contains no MDP, no selfish-mining analysis (written before Eyal-Sirer), no numbers to reproduce.
Relevance: conceptual only. It supports the "disruption (Goldfinger) objective" next to revenue: an attacker motivated by harming the chain rather than profit, which is the objective the gist asks for (reorgs of 11+). The pcmdp disruption objective is that formalised for one attacker with a fixed reorg-depth threshold; nothing to port.

## 7. Carlsten, Kalodner, Weinberg, Narayanan (CCS 2016), and the CITP blog
Sources: PDF `cs.princeton.edu/~arvindn/publications/mining_CCS.pdf` (fetched; extended version noted in the text, figures updated); blog `freedom-to-tinker.com/2016/10/21/bitcoin-is-unstable-without-the-block-reward/` (fetched). **Code: github.com/citp/mining_simulator** (58 stars, last push 2021-12-13, licence NOASSERTION; API metadata only, not cloned, not run).
Model [read]: transaction-fee-only regime: reward of a block is exponentially distributed (uniform fee arrival), so forking a "wealthy" block is profitable; equilibria involve "undercutting" (taking part of the fees; a Lambert-W miner function-fork(f) at equilibrium, Theorem 5.1 with y<=1/2, 2y - ln y >= 2), non-equilibria in some regimes; Section 6 shows selfish mining is slightly better under fees (and profitable for arbitrarily small hash power with arbitrarily poor connectivity). Analytic claims were checked in their own simulator.
Relevance: their reward model needs a different reward (variable per block) and state (unclaimed fees), which pcmdp does not have; Monero has a tail emission (minimum block reward 0.6 XMR), so the fee-only regime is not current, but block reward variance via fees is relevant to Aumayr-style splitting and to Subchains. Low priority. Useful as a second, independent simulator to cross-check a "mining on forked blocks" Monte Carlo, not to port.

## Cross-reference: numbers pcmdp can be tested against
Revenue (Nakamoto, first-seen ties, relative revenue, gamma as stated):
1. SSZ Table 2 (gamma=0), all rows above: eps-OPT 0.33705, 0.37077, 0.42600, 0.48866, 0.56808, 0.66891, 0.80172 (large caps; T=95 in SSZ); upper bounds 0.33707, 0.37079, 0.42604, 0.48904, 0.57226, 0.70109, 0.90476 (needs the over-paying award; at alpha 0.475 pcmdp's capped optimum will sit well below 0.90476 unless caps are huge).
2. SM1 fixed-policy values (gamma=0): 0.36651, 0.42118, 0.48372, 0.55801, 0.65177, 0.78255 for alpha 0.35...0.475 (SSZ 5-digit rounding differs by 1e-5; closed form [derived]). Other gamma: 0.25 at (0.25, 0.5), 0.5256 at (0.4, 0.5), 0.6806 at (0.45, 0.5), 0.5674 at (0.4, 1), 0.5591 at (0.4, 0.9).
3. Thresholds: SM1 (1-g)/(3-2g) (1/3, 0.30, 1/4, 1/6, 0 at g = 0, .25, .5, .75, 1); optimum ~0.329 at gamma=0 (SSZ via Bahrani-Weinberg, 32.9%); uniform tie-breaking at gamma=1/2 0.2321 (SSZ; already tested through Zhang-Preneel); honest threshold 0.25 for first-seen gamma=1/2.
4. Optimal policy grids: SSZ Table 3 (alpha=0.45, gamma=0.5, a,h<=8), Table 4 (alpha=0.35, gamma=0), Table 5 (uniform tie-breaking alpha=0.25).
5. gamma=1: optimum equals alpha/(1-alpha) (SSZ Prop. 1; at 0.4: 0.6667; Bahrani-Weinberg Theorem 1 too).
6. Nayak: best stubborn at (alpha=0.4, gamma=0.9) is ~0.652 (derived from "63% over honest", "23% over selfish"); T2 never beats T1; stubborn <= SSZ optimum everywhere.
7. Bahrani-Weinberg: undetectable profitable strategy exists for alpha >= 0.382 against ties-against-attacker (not a revenue value; a threshold in a constrained model).
8. RL survey thresholds are for other protocols (Ethereum 0.17 etc.); only the Bitcoin 0.25 (gamma=1/2, first-seen) and 0.21 (half small compliant miners) are comparable, the latter needs a compliant-miner model pcmdp lacks.

## Gaps, in priority order
1. SSZ over-paying upper bound for Nakamoto (+ Table 2 test): small change, makes the cap-vs-optimum question in the Aumayr disagreement a direct comparison against a published method.
2. Test SM1 closed form and Nayak L/F/T1 presets as fixed policies, including their reorg profiles under the disruption objective.
3. PTO/RL for the largest state spaces; code exists (pto repo, MIT).
4. Detectability metric (Bahrani-Weinberg) as an extra output; fee-variance models (Carlsten) skipped.
