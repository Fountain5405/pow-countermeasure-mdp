"""Subchains (weak blocks), incentive-compatibility MDP, with reorg tags.

Port of Zhang & Preneel's MATLAB MDP (PoWSecurity, "Incentive Compatibility/
Subchains", GPL-3.0): Init.m (parameters), InitStates.m (transitions),
InitStnum.m / st2stnum.m / stnum2st.m (state numbering, here `_canon`).
Rewards: reference Rs is `reward`, Rs + Rh is `difficulty_contribution`, so
the solvers' ratio is the reference's rou = Rs / (Rs + Rh).

Every mined unit is a block with probability 1/ratio (Ratio_w2b, the
reference's `mtimes`) and a weak block otherwise: Phb = (1-alpha)/ratio,
Phi = (1-alpha)(ratio-1)/ratio, Pab = alpha/ratio, Pai = alpha(ratio-1)/ratio
(InitStates.m:12-15). Weak blocks pay nothing but count as chain weight.

State (a, h, dw, fork, last), as stnum2st.m:
  a     attacker blocks (not weak blocks) since the fork, 0..max_fork;
  h     honest blocks since the fork, 0..max_fork;
  dw    weight difference attacker - honest in units, min_dw..max_dw;
  fork  Irrelevant, Relevant, Active; only kept for dw in 0..3, else Irrelevant
        (stnum2st.m:13-60);
  last  the attacker chain's last min(dw, 3) units as bits, 1 = block, newest
        lowest; kept for dw >= 1 only, mod 2^dw for dw <= 3, mod 8 above.
Actions: Illegal (pto), Adopt, Override, Match, Wait (InitStates.m:9).
The reference's illegal actions (Rh = 10000 into the base state) are pto's
final state with the error penalty here. Illegal when the state is not
canonical (the reference has no number for it).

The initial state is the reference's baseStateHi (0,0,-1): every policy that
ever adopts returns to it (Howard needs a recurrent start; the reference has none).

Reference quirk kept: baseStateAb = st2stnum(1,0,1,irrelevant,0) has last = 0
although its unit is a block (InitStates.m:11). The bit can only matter when
no honest unit has arrived since the fork, when no match is possible and
override never keeps unit 1, so it does not change any value.

Reorg tags (fork choice: the chain with more weight in units wins; the depth
is counted in BLOCKS, h is the number of honest blocks since the fork):
- Override with h > 0 (InitStates.m:68-106): the honest chain loses h blocks
  (majority, depth h) on all four successors. Weak blocks it loses count 0.
- Wait/Match with an honest unit on the attacker's branch (prob. gamma of the
  honest units, :135-149 and :195-210): the honest chain loses h blocks
  (majority, depth h).
- Wait in an Active fork or Match with an honest unit on the honest branch
  (prob. 1-gamma): the gamma share that followed the attacker's published
  units drops them: minority, depth a - blocksInLast, the attacker blocks the
  Match published (not h: chain weight counts weak blocks, so the two branches
  have equal weight but not equal block counts). Depth 0 (only weak blocks
  published) is no tag. None when gamma = 0.
- Adopt in an Active fork: the same minority drop, depth a - blocksInLast
  (none when gamma = 0 or only weak blocks were published).

(normalization "time" applies to the disruption objective only; revenue is unaffected.)
normalization "time": every legal transition mines exactly one unit, a strong
block with probability 1/ratio (Init.m: mtimes = "the inblock difficulty
threshold"; InitStates.m:12-15), so units arrive ratio times as fast as
strong blocks and a unit lasts 1/ratio strong-block intervals. Illegal: 0.
Rates are then per strong-block interval.

What the state cannot determine: once the attacker's lead passes dw = 3 during
an Active fork the reference forgets the fork (fork = Irrelevant for dw > 3),
so the gamma share it modelled is dropped from then on; no tag is produced
for that (the reference itself pays no match reward there). Also the
reference publishes all but three units on override, so for dw > 4 the
attacker's published depth is larger than a minimal override would be; the
majority depth h is unaffected.

Limitations: minority reorgs are undercounted after the fork is forgotten at
dw > 3 (accepted: the disruption objective counts majority reorgs only, so
revenue and disruption values are unaffected). The three-unit truncation
(at most three private units are tracked) and the publish-all-but-three
override come from the reference and favour the compliant miners.
"""
from __future__ import annotations

from typing import Tuple

from .reorgs import MAJORITY, MINORITY, TaggedModel, TaggedTransitions
from .vendor.blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from .vendor.blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from .vendor.blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .vendor.blockchain_mdps.base.blockchain_model import BlockchainModel


def _blocks(last: int) -> int:
    """blocksInLast (InitStates.m:93): number of 1 bits in the three low bits."""
    return (last // 4) + ((last % 4) // 2) + (last % 2)


class _OutOfRange(Exception):
    pass


class SubchainsModel(TaggedModel):
    def __init__(self, alpha: float, gamma: float, ratio: int = 2, max_fork: int = 20,
                 min_dw: int = -5, max_dw: int = 20, objective: str = "revenue",
                 min_depth: int = 11, weight: str = "count", normalization: str = "main_chain"):
        if normalization not in ("main_chain", "time"):
            raise ValueError("normalization must be main_chain or time")
        self.normalization = normalization
        self.time_rates = normalization == "time" and objective == "disruption"
        if max_dw < 4 or min_dw > -1 or max_fork < 4:
            raise ValueError("need max_dw >= 4, min_dw <= -1, max_fork >= 4")
        if ratio < 1:
            raise ValueError("ratio (Ratio_w2b) must be at least 1")
        self.init_objective(objective, min_depth, weight)
        self.alpha = alpha
        self.gamma = gamma
        self.ratio = int(ratio)
        self.max_fork = int(max_fork)
        self.min_dw = int(min_dw)
        self.max_dw = int(max_dw)
        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Adopt', 'Override', 'Match', 'Wait'])
        super().__init__()

    def _args(self) -> tuple:
        return (self.alpha, self.gamma, self.ratio, self.max_fork, self.min_dw, self.max_dw)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self._args()}, {self.objective}, {self.min_depth}, {self.weight}, {self.normalization})"

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, self._args() + (self.objective, self.min_depth, self.weight,
                                               self.normalization)

    def get_state_space(self):
        underlying = MultiDimensionalDiscreteSpace((0, self.max_fork), (0, self.max_fork),
                                                   (self.min_dw, self.max_dw), self.Fork, (0, 7))
        return DefaultValueSpace(underlying, self.get_final_state())

    def get_action_space(self):
        return DiscreteSpace(self.Action)

    def get_initial_state(self):
        return 0, 0, -1, self.Fork.Irrelevant, 0

    def get_final_state(self):
        return -1, -1, 0, self.Fork.Irrelevant, 0

    def _canon(self, a, h, dw, fork, last) -> tuple:
        """st2stnum.m: the state the reference numbers (a, h, dw, fork, last) as."""
        if not (0 <= a <= self.max_fork and 0 <= h <= self.max_fork and self.min_dw <= dw <= self.max_dw):
            raise _OutOfRange
        if dw < 0:
            fork, last = 0, 0
        elif dw == 0:
            last = 0
        elif dw <= 3:
            last %= 2 ** dw
        else:
            fork, last = 0, last % 8
        return a, h, dw, self.Fork(fork), last

    def get_tagged_transitions(self, state, action, check_valid: bool = True) -> TaggedTransitions:
        t = TaggedTransitions()
        if state == self.final_state:
            t.add(self.final_state, probability=1)
            return t
        try:
            self._fill(t, state, action)
        except _OutOfRange:
            t = TaggedTransitions()
            t.add(self.final_state, probability=1, reward=self.error_penalty)
        if self.time_rates:
            for nxt in t.difficulty_contributions:
                t.difficulty_contributions[nxt] = 0.0 if nxt == self.final_state else 1.0 / self.ratio
        return t

    def _illegal(self, t):
        t.add(self.final_state, probability=1, reward=self.error_penalty)

    def _fill(self, t: TaggedTransitions, state, action) -> None:
        a, h, dw, fork, last = state
        if self._canon(*state) != tuple(state):
            return self._illegal(t)
        m = self.ratio
        al, ga = self.alpha, self.gamma
        Phb, Phi = (1 - al) / m, (1 - al) * (m - 1) / m
        Pab, Pai = al / m, al * (m - 1) / m
        Irr, Rel, Act = self.Fork.Irrelevant, self.Fork.Relevant, self.Fork.Active
        A = self.Action
        c = self._canon

        def add(nxt, p, rs=0.0, rh=0.0, reorgs=()):
            if p > 0:
                t.add(nxt, probability=p, reward=rs, difficulty_contribution=rs + rh,
                      allow_merging=True, reorgs=reorgs)

        if action is A.Illegal:
            return t.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        if action is A.Adopt:
            # InitStates.m:20-23, 52-62; Rh = h on all four successors (:84)
            # from an Active fork the gamma share on the published units drops them
            blk = _blocks(last)
            drop = [(MINORITY, a - blk)] if fork is self.Fork.Active and ga > 0 else []
            add(c(0, 1, -1, Irr, 0), Phb, rh=h, reorgs=drop)
            add(c(0, 0, -1, Irr, 0), Phi, rh=h, reorgs=drop)
            add(c(1, 0, 1, Irr, 0), Pab, rh=h, reorgs=drop)
            add(c(0, 0, 1, Irr, 0), Pai, rh=h, reorgs=drop)
            return

        blocks = _blocks(last)
        if a < blocks:  # InitStates.m:86-88: everything but adopt is illegal
            return self._illegal(t)

        if action is A.Override:
            if dw <= 0:
                return self._illegal(t)
            left = min(3, dw - 1)                       # :92
            new_last = last % (2 ** left)
            a_left = _blocks(new_last)
            rs = a - a_left
            tag = [(MAJORITY, h)]
            add(c(a_left, 1, left - 1, Rel, new_last), Phb, rs=rs, reorgs=tag)        # :101
            add(c(a_left, 0, left - 1, Rel, new_last), Phi, rs=rs, reorgs=tag)        # :107
            add(c(a_left + 1, 0, left + 1, Irr, new_last * 2 + 1), Pab, rs=rs, reorgs=tag)  # :113
            add(c(a_left, 0, left + 1, Irr, new_last * 2), Pai, rs=rs, reorgs=tag)    # :119
            return

        if action is A.Wait:
            if not (a < self.max_fork and h < self.max_fork and self.min_dw < dw < self.max_dw):
                return self._illegal(t)         # :125
            if fork is not Act:
                add(c(a, h + 1, dw - 1, Rel, last), Phb)                              # :130
                add(c(a, h, dw - 1, Rel, last), Phi)                                  # :135
                add(c(a + 1, h, dw + 1, Irr, last * 2 + 1), Pab)                      # :140
                add(c(a, h, dw + 1, Irr, last * 2), Pai)                              # :145
                return
            self._race(t, add, state, blocks, Act)
            return

        if action is A.Match:
            # :190: feasible only after an honest unit, dw 0..3, room for one more unit
            if fork is Rel and 0 <= dw <= 3 and a + 1 <= self.max_fork and h + 1 <= self.max_fork:
                self._race(t, add, state, blocks, Act)
                return
            return self._illegal(t)

    def _race(self, t, add, state, blocks, Act) -> None:
        """The six outcomes of a unit mined while the attacker's chain is published
        up to the honest weight (Wait in an Active fork :153-186, Match :190-232)."""
        a, h, dw, fork, last = state
        m = self.ratio
        al, ga = self.alpha, self.gamma
        Phb, Phi = (1 - al) / m, (1 - al) * (m - 1) / m
        Pab, Pai = al / m, al * (m - 1) / m
        Rel = self.Fork.Relevant
        c = self._canon
        published = a - blocks                          # blocks the attacker has published
        drop = [(MINORITY, published)] if ga > 0 else []
        win = [(MAJORITY, h)]
        add(c(a, h + 1, dw - 1, Rel, last), Phb * (1 - ga), reorgs=drop)
        add(c(a, h, dw - 1, Rel, last), Phi * (1 - ga), reorgs=drop)
        add(c(blocks, 1, dw - 1, Rel, last), Phb * ga, rs=published, reorgs=win)
        add(c(blocks, 0, dw - 1, Rel, last), Phi * ga, rs=published, reorgs=win)
        add(c(a + 1, h, dw + 1, Act, last * 2 + 1), Pab)
        add(c(a, h, dw + 1, Act, last * 2), Pai)
