"""Zhang & Preneel's three tie-breaking chain-quality MDPs, with reorg tags.

Ported from PoWSecurity/Chain Quality/{UniformTieBreaking,UDTieBreaking,
SmallestHashTieBreaking} (Zhang & Preneel 2019, "Lay Down the Common Metrics",
MATLAB, GPL-3.0); file:line references are to those folders' InitStates.m.
These protocols change only how an equal-length fork is resolved.

Rewards: the reference carries Rs (attacker) and Rh (honest) and bisects on
rou = Rs / (Rs + Rh). Here reward = Rs and difficulty_contribution = Rs + Rh
on every transition, so the solvers' ratio is the reference's rou (the
attacker's share of main-chain blocks). The honest side is paid h at Adopt
and nothing elsewhere, as in the reference.

Every action ends with one newly mined block already drawn (the reference
samples it inside the action), so normalization "time" = 1 per transition.

Reorg tags (depth = honest public blocks removed; the state's h is exactly
the honest blocks since the common ancestor, which each publish resets):
- UDTB, SHTB: compliant miners resolve ties deterministically and all follow
  the same chain, so every publish that wins removes h honest blocks from the
  chain *all* of them hold: (MAJORITY, h) on OverrideMore, OverrideTie, and on
  SHTB's OverrideSafe / Override. No minority reorg exists.
- UTB: Match publishes h blocks into a tie and half of the honest miners
  (gamma) follow it. As in nakamoto.py: an honest block found on the attacker
  branch (gamma(1-alpha)) removes h blocks, (MAJORITY, h); one found on the
  honest branch ((1-gamma)(1-alpha)) makes the gamma share drop the attacker's
  h published blocks, (MINORITY, h) (none when gamma is 0). Override tags
  (MAJORITY, h).
The state cannot say: with gamma = 0.5 exactly half of the honest power is on
each side of an Active tie, so "majority" for the Active branch is a
convention (inherited from nakamoto.py); and Adopt from an Active UTB state
(the attacker abandons its published tie branch, which the gamma share may
still be mining on) is untagged.
"""
from __future__ import annotations

from math import isqrt
from typing import Any, List, Tuple

import numpy as np

from .reorgs import MAJORITY, MINORITY, TaggedModel, TaggedTransitions
from .vendor.blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from .vendor.blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from .vendor.blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .vendor.blockchain_mdps.base.base_space.space import Space
from .vendor.blockchain_mdps.base.blockchain_model import BlockchainModel


class _TieBreakingModel(TaggedModel):
    """Shared plumbing: objective, final state, normalization, honest policy."""

    ACTIONS: List[str] = []

    def _setup(self, alpha, max_fork, objective, min_depth, weight, normalization):
        if normalization not in ("main_chain", "time"):
            raise ValueError("normalization must be main_chain or time")
        self.alpha = alpha
        self.max_fork = max_fork
        self.normalization = normalization
        self.init_objective(objective, min_depth, weight)
        self.Action = self.create_int_enum("Action", self.ACTIONS)

    def get_action_space(self) -> Space:
        return DiscreteSpace(self.Action)

    def _finish(self, transitions: TaggedTransitions, action) -> TaggedTransitions:
        if self.normalization == "time":
            mined = action is not self.Action.Illegal
            for next_state in transitions.difficulty_contributions:
                transitions.difficulty_contributions[next_state] = (
                    1.0 if mined and next_state != self.final_state else 0.0)
        return transitions

    def _illegal(self, transitions: TaggedTransitions, penalty: float = None) -> TaggedTransitions:
        transitions.add(self.final_state, probability=1,
                        reward=self.error_penalty if penalty is None else penalty)
        return transitions

    def _honest_action(self, state) -> Any:
        raise NotImplementedError

    def build_honest_policy(self) -> tuple:
        """Adopt when behind, override when ahead, wait when level: earns alpha."""
        policy = np.zeros(self.state_space.size, dtype=int)
        for i in range(1, self.state_space.size):
            policy[i] = self._honest_action(self.state_space.index_to_element(i))
        return tuple(policy)


class UniformTieBreakingModel(_TieBreakingModel):
    """UTB: a tie splits the honest power, gamma on the attacker's branch (ref. 0.5).

    State (a, h, fork): attacker's private blocks, honest public blocks since
    the fork, fork 0 inactive / 1 active (a tie is published). Actions as in
    the reference: adopt, override, match, wait (UniformTieBreaking/InitStates.m).
    """

    ACTIONS = ["Illegal", "Adopt", "Override", "Match", "Wait"]

    def __init__(self, alpha: float, max_fork: int = 80, gamma: float = 0.5, objective: str = "revenue",
                 min_depth: int = 11, weight: str = "count", normalization: str = "main_chain"):
        self.gamma = gamma
        self._setup(alpha, max_fork, objective, min_depth, weight, normalization)
        super().__init__()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.alpha}, {self.max_fork}, {self.gamma}, {self.objective}, "
                f"{self.min_depth}, {self.weight}, {self.normalization})")

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.max_fork, self.gamma, self.objective, self.min_depth,
                                self.weight, self.normalization)

    def get_state_space(self) -> Space:
        return DefaultValueSpace(
            MultiDimensionalDiscreteSpace((0, self.max_fork), (0, self.max_fork), (0, 1)), self.get_final_state())

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 1, 0  # the reference's baseState

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, 0

    def _honest_action(self, state):
        a, h, _ = state
        return self.Action.Adopt if h > a else self.Action.Override if a > h else self.Action.Wait

    def get_tagged_transitions(self, state, action, check_valid: bool = True) -> TaggedTransitions:
        t = TaggedTransitions()
        if state == self.final_state:
            t.add(self.final_state, probability=1)
            return t
        a, h, fork = state
        A, L, al, g = self.Action, self.max_fork, self.alpha, self.gamma

        if action is A.Illegal:
            return self._illegal(t, self.error_penalty / 2)

        if action is A.Adopt:  # lines 62-68, 83-84: honest gets h, next block drawn from the base states
            t.add((0, 1, 0), probability=1 - al, difficulty_contribution=h)
            t.add((1, 0, 0), probability=al, difficulty_contribution=h)

        elif action is A.Override:  # lines 86-96
            if a > h:
                tags = [(MAJORITY, h)]
                t.add((a - h, 0, 0), probability=al, reward=h + 1, difficulty_contribution=h + 1, reorgs=tags)
                t.add((a - h - 1, 1, 0), probability=1 - al, reward=h + 1, difficulty_contribution=h + 1,
                      reorgs=tags)
            else:
                return self._illegal(t)

        elif action is A.Match:  # lines 122-135
            if a >= h and h > 0 and a + 1 <= L and h + 1 <= L:
                self._race(t, a, h)
            else:
                return self._illegal(t)

        elif action is A.Wait:  # lines 98-119
            if fork == 0 and a + 1 <= L and h + 1 <= L:
                t.add((a + 1, h, 0), probability=al)
                t.add((a, h + 1, 0), probability=1 - al)
            elif fork == 1 and a > h > 0 and a + 1 <= L and h + 1 <= L:
                self._race(t, a, h)
            else:
                return self._illegal(t)
        return self._finish(t, action)

    def _race(self, t: TaggedTransitions, a: int, h: int) -> None:
        """h blocks are published into a tie; the next block decides (lines 107-118 / 124-134)."""
        al, g = self.alpha, self.gamma
        t.add((a + 1, h, 1), probability=al)
        # honest block on the attacker's branch: its h blocks win, the honest h are orphaned
        t.add((a - h, 1, 0), probability=g * (1 - al), reward=h, difficulty_contribution=h,
              reorgs=[(MAJORITY, h)])
        # honest block on the honest branch: the gamma share drops the attacker's h blocks
        t.add((a, h + 1, 0), probability=(1 - g) * (1 - al), reorgs=[(MINORITY, h)] if g > 0 else [])


class UDTieBreakingModel(_TieBreakingModel):
    """UDTB: unpredictable deterministic tie-breaking (UDTieBreaking/InitStates.m).

    State (a, h, tie): tie 1 = aWin (the attacker's tip beats the public tip
    in a tie), 0 = aLose; meaningful when a >= h > 0. Every new public tip
    gets a fresh coin. Actions: adopt, overrideTie, overrideMore, wait.
    """

    ACTIONS = ["Illegal", "Adopt", "OverrideTie", "OverrideMore", "Wait"]

    def __init__(self, alpha: float, max_fork: int = 160, objective: str = "revenue", min_depth: int = 11,
                 weight: str = "count", normalization: str = "main_chain"):
        self._setup(alpha, max_fork, objective, min_depth, weight, normalization)
        super().__init__()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.alpha}, {self.max_fork}, {self.objective}, "
                f"{self.min_depth}, {self.weight}, {self.normalization})")

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.max_fork, self.objective, self.min_depth, self.weight,
                                self.normalization)

    get_state_space = UniformTieBreakingModel.get_state_space
    get_initial_state = UniformTieBreakingModel.get_initial_state
    get_final_state = UniformTieBreakingModel.get_final_state

    def _honest_action(self, state):
        a, h, _ = state
        return self.Action.Adopt if h > a else self.Action.OverrideMore if a > h else self.Action.Wait

    def _coin(self, t, a, h, p, reward=0, diff=0, tags=()):
        """Next state (a, h, tie) with a fresh tie coin when it is meaningful (a >= h > 0)."""
        if a == 0 or a < h:
            t.add((a, h, 0), probability=p, reward=reward, difficulty_contribution=diff, reorgs=tags)
        else:
            t.add((a, h, 0), probability=p / 2, reward=reward, difficulty_contribution=diff, reorgs=tags)
            t.add((a, h, 1), probability=p / 2, reward=reward, difficulty_contribution=diff, reorgs=tags)

    def get_tagged_transitions(self, state, action, check_valid: bool = True) -> TaggedTransitions:
        t = TaggedTransitions()
        if state == self.final_state:
            t.add(self.final_state, probability=1)
            return t
        a, h, tie = state
        A, L, al = self.Action, self.max_fork, self.alpha

        if action is A.Illegal:
            return self._illegal(t, self.error_penalty / 2)

        if action is A.Adopt:  # lines 58-65, 80-81
            t.add((0, 1, 0), probability=1 - al, difficulty_contribution=h)
            t.add((1, 0, 0), probability=al, difficulty_contribution=h)

        elif action is A.OverrideMore:  # lines 83-102: publish h+1, honest h orphaned
            if a > h:
                tags = [(MAJORITY, h)]
                t.add((a - h, 0, 0), probability=al, reward=h + 1, difficulty_contribution=h + 1, reorgs=tags)
                # a-h-1 == 0: (0, 1, .) has no tie bit; else the new honest tip gets a coin (lines 92-101)
                self._coin(t, a - h - 1, 1, 1 - al, h + 1, h + 1, tags)
            else:
                return self._illegal(t)

        elif action is A.OverrideTie:  # lines 104-123: publish h, win the tie
            if a >= h and tie == 1 and a - h + 1 <= L:
                tags = [(MAJORITY, h)]
                t.add((a - h + 1, 0, 0), probability=al, reward=h, difficulty_contribution=h, reorgs=tags)
                self._coin(t, a - h, 1, 1 - al, h, h, tags)
            else:
                return self._illegal(t)

        elif action is A.Wait:  # lines 125-165
            if a + 1 <= L and h + 1 <= L:
                if a > h:
                    t.add((a + 1, h, tie), probability=al)
                    self._coin(t, a, h + 1, 1 - al)
                elif a == h:
                    t.add((a + 1, h, tie), probability=al)
                    t.add((a, h + 1, tie), probability=1 - al)
                else:
                    t.add((a + 1, h, 0), probability=al / 2)
                    t.add((a + 1, h, 1), probability=al / 2)
                    t.add((a, h + 1, 0), probability=1 - al)
            else:
                return self._illegal(t)
        return self._finish(t, action)


class _CompactSpace(Space):
    """SHTB states (a, h, tie, slot1, slot2) in the reference's numbering (st2stnum.m / stnum2st.m).

    a < h: slot1 = hash region of the public tip (htop); a == h > 0: tie;
    a == h + 1: tie, slot1 = region of the last attacker block; a > h + 1:
    slot1, slot2 = regions of the last two attacker blocks. Unused fields are
    -1 (tie 0). Indices are the reference's minus one.
    """

    def __init__(self, max_fork: int, regions: int):
        self.L, self.R = max_fork, regions
        self.t_small = max_fork * (max_fork + 1) // 2 * regions
        self.t_eq = self.t_small + 2 * max_fork
        self.t_one = self.t_eq + 2 * max_fork * regions
        super().__init__()

    def _calc_dimension(self) -> int:
        return 5

    def _calc_size(self) -> int:
        return self.t_one + self.L * (self.L - 1) // 2 * self.R * self.R

    def element_to_index(self, e) -> int:
        a, h, tie, s1, s2 = e
        R = self.R
        if a < h:
            return (h * (h - 1) // 2 + a) * R + s1
        if a == h:
            return self.t_small + (a - 1) * 2 + tie
        if a == h + 1:
            return self.t_eq + ((a - 1) * 2 + tie) * R + s1
        return self.t_one + (((a - 2) * (a - 1) // 2 + h) * R + s1) * R + s2

    def index_to_element(self, index: int) -> tuple:
        if index < 0 or index >= self.size:
            raise ValueError("Bad index")
        R = self.R
        if index < self.t_small:
            s1, n = index % R, index // R
            h = (1 + isqrt(8 * n + 1)) // 2
            return n - h * (h - 1) // 2, h, 0, s1, -1
        if index < self.t_eq:
            n = index - self.t_small
            a = n // 2 + 1
            return a, a, n % 2, -1, -1
        if index < self.t_one:
            n = index - self.t_eq
            s1, n = n % R, n // R
            a = n // 2 + 1
            return a, a - 1, n % 2, s1, -1
        n = index - self.t_one
        s2, n = n % R, n // R
        s1, n = n % R, n // R
        a = (1 + isqrt(8 * n + 1)) // 2 + 1
        return a, n - (a - 2) * (a - 1) // 2, 0, s1, s2

    def transform_element(self, element: Any) -> Any:
        return element

    def enumerate_dimension(self, coordinate: int) -> List[int]:
        raise NotImplementedError


def _canon(a: int, h: int, tie: int = 0, s1: int = -1, s2: int = -1) -> tuple:
    """The one tuple per state: drop the fields the reference's encoding drops."""
    if a < h:
        return a, h, 0, s1, -1
    if a == h:
        return a, h, tie, -1, -1
    if a == h + 1:
        return a, h, tie, s1, -1
    return a, h, 0, s1, s2


class SmallestHashTieBreakingModel(_TieBreakingModel):
    """SHTB: a tie goes to the smaller hash; hashes are split into `regions` equal regions.

    Same region = the public tip wins (favours the honest side), so the
    attacker's value is a lower bound and the profitable threshold an upper
    bound (SmallestHashTieBreaking/InitStates.m). Actions: adopt, overrideSafe
    (publish h+1, or everything when that is all there is / a tie is won),
    override (publish h, a_ = h + 1 with the tie won), wait.
    """

    ACTIONS = ["Illegal", "Adopt", "OverrideSafe", "Override", "Wait"]

    def __init__(self, alpha: float, max_fork: int = 40, regions: int = 15, objective: str = "revenue",
                 min_depth: int = 11, weight: str = "count", normalization: str = "main_chain"):
        self.regions = regions
        self._setup(alpha, max_fork, objective, min_depth, weight, normalization)
        super().__init__()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.alpha}, {self.max_fork}, {self.regions}, {self.objective}, "
                f"{self.min_depth}, {self.weight}, {self.normalization})")

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.max_fork, self.regions, self.objective, self.min_depth,
                                self.weight, self.normalization)

    def get_state_space(self) -> Space:
        return DefaultValueSpace(_CompactSpace(self.max_fork, self.regions), self.get_final_state())

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 1, 0, 0, -1  # the reference's baseStateH(1)

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, 0, -1, -1

    def _honest_action(self, state):
        a, h = state[0], state[1]
        return self.Action.Adopt if h > a else self.Action.OverrideSafe if a > h else self.Action.Wait

    def get_tagged_transitions(self, state, action, check_valid: bool = True) -> TaggedTransitions:
        t = TaggedTransitions()
        if state == self.final_state:
            t.add(self.final_state, probability=1)
            return t
        a, h, tie, s1, s2 = state
        A, L, R, al = self.Action, self.max_fork, self.regions, self.alpha
        ph, ps = (1 - al) / R, al / R
        if a < h:
            htop = s1
        else:
            atop1, atop2 = s1, s2

        def add(nxt, p, reward=0, diff=0, tags=()):
            if p > 0:  # zero-probability branches (a hash cannot be smaller than region 0) are dropped
                t.add(nxt, probability=p, reward=reward, difficulty_contribution=diff, reorgs=tags)

        def base(reward=0, diff=0, tags=()):
            for r in range(R):
                add(_canon(0, 1, 0, r), ph, reward, diff, tags)
                add(_canon(1, 0, 0, r), ps, reward, diff, tags)

        if action is A.Illegal:
            return self._illegal(t, self.error_penalty / 2)

        if action is A.Adopt:  # lines 57-71, 92
            base(0, h)

        elif action is A.OverrideSafe:  # lines 96-147
            tags = [(MAJORITY, h)]
            if a == h + 1 or (a == h and tie == 1):  # publish everything (lines 100-109)
                base(a, a, tags)
            elif a > h + 1:  # publish h+1 blocks, keep the rest private (lines 110-146)
                for r in range(R):
                    add(_canon(a - h, 0, 0, r, atop1), ps, h + 1, h + 1, tags)
                if a == h + 2:  # one block left (atop1) competes with the new honest tip
                    add(_canon(1, 1, 1), ph * (R - 1 - atop1), h + 1, h + 1, tags)
                    add(_canon(1, 1, 0), ph * (1 + atop1), h + 1, h + 1, tags)
                elif a == h + 3:  # two left: the lower one (atop2) is the competitor
                    add(_canon(2, 1, 1, atop1), ph * (R - 1 - atop2), h + 1, h + 1, tags)
                    add(_canon(2, 1, 0, atop1), ph * (1 + atop2), h + 1, h + 1, tags)
                else:
                    add(_canon(a - h - 1, 1, 0, atop1, atop2), 1 - al, h + 1, h + 1, tags)
            else:
                return self._illegal(t)

        elif action is A.Override:  # lines 151-171: a == h+1 with the tie won, publish h only
            if a == h + 1 and tie == 1:
                tags = [(MAJORITY, h)]
                add(_canon(1, 1, 1), ph * (R - 1 - atop1), h, h, tags)
                add(_canon(1, 1, 0), ph * (1 + atop1), h, h, tags)
                for r in range(R):
                    add(_canon(2, 0, 0, r, atop1), ps, h, h, tags)
            else:
                return self._illegal(t)

        elif action is A.Wait:  # lines 174-233
            if a + 1 <= L and h + 1 <= L:
                if a <= h:  # honest block, new public tip in a uniform region
                    for r in range(R):
                        add(_canon(a, h + 1, 0, r), ph)
                elif a == h + 1:
                    add(_canon(a, h + 1, 1), ph * (R - 1 - atop1))
                    add(_canon(a, h + 1, 0), ph * (1 + atop1))
                elif a == h + 2:
                    add(_canon(a, h + 1, 1, atop1), ph * (R - 1 - atop2))
                    add(_canon(a, h + 1, 0, atop1), ph * (1 + atop2))
                else:
                    add(_canon(a, h + 1, 0, atop1, atop2), 1 - al)
                if a < h - 1:  # attacker block, still behind
                    add(_canon(a + 1, h, 0, htop), al)
                elif a == h - 1:  # catches up: wins the tie iff its region is below htop
                    add(_canon(a + 1, h, 1), ps * htop)
                    add(_canon(a + 1, h, 0), ps * (R - htop))
                elif a == h:
                    for r in range(R):
                        add(_canon(a + 1, h, tie, r), ps)
                else:
                    for r in range(R):
                        add(_canon(a + 1, h, 0, r, atop1), ps)
            else:
                return self._illegal(t)
        return self._finish(t, action)
