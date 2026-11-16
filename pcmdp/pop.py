"""Publish or Perish (PoP, Zhang & Preneel) chain-quality MDP with reorg tags.

Port of the MATLAB MDP in github.com/... PoWSecurity/Chain Quality/PublishOrPerish
(GPL-3.0; Zhang & Preneel, "Lay Down the Common Metrics" 2019, section IV).
File:line cites below are to InitStates.m unless named.

State (Bh, Bs, Dw, luck, last, published), as in st2stnum.m:
  Bh, Bs     honest / selfish blocks since the fork;
  Dw         published honest weight minus published selfish weight (weight of
             in-time blocks; 0 means the published chains tie);
  luck       the Bh+1-th selfish block was mined after the Bh-th honest block
             (publishing it would earn two weight points);
  last       0 = the last block was honest (lastH), 1 = selfish (lastS);
  published  selfish blocks already public.
Actions: adopt, override, match, even, hide (k = `k` is the fail-safe: the
attacker may override with a lighter chain when k blocks ahead).

Rewards (PORTING.md): reward = Rs (attacker), difficulty = Rs + Rh, so the
solvers' ratio is the reference's rou = Rs / (Rs + Rh).

Departures from the MATLAB (each one a place the .m file leans on an accident):
- States whose Dw is outside 0 .. Bh+1 (the `continue` test at line 52, which
  allows only adopt) are all mapped to one dead value Dw = max_b + 2. The .m
  file lets such Dw (and negative Dw, from minDw or newDw) overflow into the
  neighbouring digits of the state number; the aliased state is adopt-only
  too except possibly at the race cap, where our value differs.
- Line 142 (match, "publish everything") uses `honestLuck` without setting it
  in that iteration: MATLAB carries the value over from an earlier state.
  We use the rule of line 116: 2 if the Bh-th selfish block is public
  (Bs >= Bh) and Bh > 0, else 1. `stale_honest_luck` overrides it (1 or 2)
  for sensitivity runs.
- The 0.5 tie split is the `tie` argument (default 0.5).

Reorg tags (state and action determine the depth except where noted):
- override: the honest chain loses its Bh blocks (majority, depth Bh).
- match, honest block on the attacker's branch (prob. tie * (1-alpha)): the
  attacker's published blocks win, the honest chain loses Bh blocks (majority).
- match, honest block on the honest branch (prob. (1-tie) * (1-alpha)): the
  tie share that followed the attacker's matchPublishUntil published blocks
  drops them (minority, depth matchPublishUntil). Not determined by the
  state, an assumption: the model does not record how many honest miners were on the
  attacker's branch, nor for how many of its blocks; we assume the tie share
  followed the whole published branch.
- even, hide, the "publish everything" match, adopt: honest miners stay on
  their own chain, nothing is reorganised publicly.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .reorgs import MAJORITY, MINORITY, TaggedModel, TaggedTransitions
from .vendor.blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from .vendor.blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from .vendor.blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .vendor.blockchain_mdps.base.base_space.space import Space
from .vendor.blockchain_mdps.base.blockchain_model import BlockchainModel

LAST_H, LAST_S = 0, 1


class PublishOrPerishModel(TaggedModel):
    def __init__(self, alpha: float, max_b: int = 30, k: int = 3, tie: float = 0.5,
                 objective: str = "revenue", min_depth: int = 11, weight: str = "count",
                 stale_honest_luck: Optional[int] = None):
        self.alpha = alpha
        self.max_b = max_b
        self.k = k
        self.tie = tie
        self.stale_honest_luck = stale_honest_luck
        self.dead = max_b + 2
        self.init_objective(objective, min_depth, weight)
        self.Action = self.create_int_enum("Action", ["Illegal", "Adopt", "Override", "Match", "Even", "Hide"])
        super().__init__()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.alpha}, {self.max_b}, {self.k}, {self.tie}, "
                f"{self.objective}, {self.min_depth}, {self.weight}, {self.stale_honest_luck})")

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.max_b, self.k, self.tie, self.objective,
                                self.min_depth, self.weight, self.stale_honest_luck)

    def get_state_space(self) -> Space:
        m = self.max_b
        underlying = MultiDimensionalDiscreteSpace((0, m), (0, m), (0, m + 2), (0, 1), (0, 1), (0, m))
        return DefaultValueSpace(underlying, self.get_final_state())

    def get_action_space(self) -> Space:
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        return 1, 0, 1, 0, LAST_H, 0          # baseState, Init (line 14)

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, -1, -1, -1, -1

    def get_honest_revenue(self) -> float:
        return self.alpha if self.objective == "revenue" else 0.0

    def _st(self, bh: int, bs: int, dw: int, luck: int, last: int, published: int) -> tuple:
        """st2stnum's role: a state tuple, with unrepresentable Dw sent to the dead value."""
        if dw < 0 or dw > bh + 1:
            dw = self.dead
        return bh, bs, dw, luck, last, published

    def _add(self, t: TaggedTransitions, state: tuple, p: float, rs: float = 0.0, rh: float = 0.0,
             reorgs=()) -> None:
        """Add a transition; merge into an existing one with probability-weighted rewards."""
        if p <= 0:
            return
        if state in t.probabilities:
            old = t.probabilities[state]
            t.rewards[state] = (old * t.rewards[state] + p * rs) / (old + p)
            t.difficulty_contributions[state] = (old * t.difficulty_contributions[state] + p * (rs + rh)) / (old + p)
            t.add(state, probability=p, allow_merging=True, reorgs=reorgs)
        else:
            t.add(state, probability=p, reward=rs, difficulty_contribution=rs + rh, reorgs=reorgs)

    def get_tagged_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                               check_valid: bool = True) -> TaggedTransitions:
        t = TaggedTransitions()
        if state == self.final_state:
            t.add(self.final_state, probability=1)
            return t
        if action is self.Action.Illegal:
            t.add(self.final_state, probability=1, reward=self.error_penalty / 2)
            return t

        bh, bs, dw, luck, last, published = state
        psb, phb = self.alpha, 1 - self.alpha
        k, mb = self.k, self.max_b
        H, S = LAST_H, LAST_S

        if action is self.Action.Adopt:                       # lines 29-37: Rh = Bh
            self._add(t, self.get_initial_state(), phb, rh=bh)
            self._add(t, (0, 1, 0, 0, S, 0), psb, rh=bh)
            return t

        def illegal() -> TaggedTransitions:
            u = TaggedTransitions()
            u.add(self.final_state, probability=1, reward=self.error_penalty)
            return u

        # illegal / forced-adopt states, line 52
        if (published > bs - last or dw > bh + 1 or (last == H and bh == 0) or (last == S and bs == 0)
                or (luck == 1 and (bh == 0 or bs < bh or (bs == bh and last == S)))):
            return illegal()

        lucky_one, new_luck_h = -1, 0
        if luck == 1:
            if last == H:
                lucky_one = bh
            else:
                lucky_one = bh + 1
                new_luck_h = 1
        new_luck_s = 1 if (bs == bh and bh > 0) else 0
        nxt = published + 1
        if nxt < bh and last == H:
            nxt = bh
        elif nxt < bh + 1 and last == S:
            nxt = bh + 1
        min_dw = dw if bs < nxt else dw - (bs - nxt + 1 + luck)

        if action is self.Action.Override:                    # lines 86-106
            if not (min_dw < 0 or bs - bh >= k):
                return illegal()
            pu = bs + min_dw + 1
            if pu - bh > k:
                pu = bh + k
            if pu < lucky_one:
                pu = lucky_one
            if pu > bs or pu < 0 or bs - pu + 1 > mb:
                return illegal()          # not reachable in the .m file's states; guard only
            tags = [(MAJORITY, bh)]       # the honest chain loses its bh blocks
            self._add(t, (0, bs - pu + 1, 0, 0, S, 0), psb, rs=pu, reorgs=tags)
            self._add(t, (1, bs - pu, 1, 0, H, 0), phb, rs=pu, reorgs=tags)
            return t

        if min_dw <= 0:                                       # lines 108-115
            mpu = bs + min_dw
            if mpu == bh - 1 and lucky_one == bh + 1:
                mpu = bh
            elif luck == 1 and mpu == lucky_one - 1:
                mpu = -1
        else:
            mpu = -1

        if action is self.Action.Match:                       # lines 117-164
            if not (bs < mb and bh < mb):
                return illegal()
            if mpu != -1:
                honest_luck = 2 if (mpu >= bh and bh > 0) else 1
                first = (bh, bs + 1, 0, 1 if (lucky_one == bh + 1 and mpu == bh) else new_luck_s, S, mpu)
                self._add(t, first, psb)
                # honest block on the honest branch: the tie share that followed
                # the attacker's mpu published blocks drops them
                self._add(t, self._st(bh + 1, bs, honest_luck, new_luck_h, H, mpu), phb * (1 - self.tie),
                          reorgs=[(MINORITY, mpu)] if self.tie > 0 else [])
                # honest block on the attacker's branch: attacker's mpu blocks win
                temp = (1, bs - mpu, honest_luck, 0, H, 0) if bs > mpu else (1, 0, 1, 0, H, 0)
                self._add(t, self._st(*temp), phb * self.tie, rs=mpu, reorgs=[(MAJORITY, bh)])
            else:                                             # lines 165-176: publish everything
                if self.stale_honest_luck is not None:
                    honest_luck = self.stale_honest_luck
                else:
                    honest_luck = 2 if (bs >= bh and bh > 0) else 1
                self._add(t, self._st(bh, bs + 1, min_dw, new_luck_s, S, bs), psb)
                self._add(t, self._st(bh + 1, bs, min_dw + honest_luck, 0, H, bs), phb)
            return t

        if action is self.Action.Even:                        # lines 178-215
            even_pu = -1
            new_dw = 0
            if bs >= bh and dw > 0:
                if published >= bh:
                    even_pu, new_dw = published, dw
                elif last == H:
                    even_pu, new_dw = bh, dw - 1
                    if lucky_one == bh:
                        new_dw -= 1
                else:
                    even_pu, new_dw = bh, dw
            if even_pu == -1 or bs >= mb:
                return illegal()
            honest_luck = 2 if bh > 0 else 1
            self._add(t, self._st(bh, bs + 1, new_dw, 1 if lucky_one == bh + 1 else new_luck_s, S, even_pu), psb)
            self._add(t, self._st(bh + 1, bs, new_dw + honest_luck, new_luck_h, H, even_pu), phb)
            return t

        # Hide, lines 217-239
        if not (published < bh and bh >= 1 and bs < mb and bh < mb):
            return illegal()
        hide_pu = bh - 1 if bs >= bh - 1 else bs
        self._add(t, self._st(bh, bs + 1, dw, 1 if lucky_one == bh + 1 else new_luck_s, S, hide_pu), psb)
        self._add(t, self._st(bh + 1, bs, dw + 1, new_luck_h, H, hide_pu), phb)
        return t
