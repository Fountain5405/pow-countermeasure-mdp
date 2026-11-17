"""Reward Splitting (RS) and Proportional Reward Splitting (PRS), with reorg tags.

Port of Aumayr et al.'s MATLAB MDPs (proportional-reward-splitting-MDP,
incentive_compatibility/RewardSplitting and /ProportionalRewardSplitting,
GPL-3.0): InitStates.m builds the transitions, st2stnum.m / stnum2st.m the
state numbering, countBlocks.m the history popcount. Rewards: reference Rs is
`reward`, Rs + Rh is `difficulty_contribution`, so the solvers' ratio is the
reference's rou = Rs / (Rs + Rh).

State (a, h, fork, history):
  a, h     attacker's private / honest public blocks since the fork, 0..max_fork;
  fork     Irrelevant, Relevant, Active (as Nakamoto); the reference stores no
           fork for a < h (match is impossible), so such states are normalised
           to Irrelevant here too;
  history  `window - 1` bits, one per recent main-chain block still inside the
           reward window, 1 = attacker block (stnum2st.m: history).
Actions: Illegal (pto), Adopt, Wait, Match, Override1..Override{window-1}
(InitStates.m:10-11: adopt, wait, match, override with margin k = 1..timeOut-1;
the attacker publishes h + k blocks).

Transitions follow InitStates.m line for line (cited per block below). The
reference's illegal actions (Rh = 10000 into the base state) are pto's
illegal actions here (final state, error penalty).

Reorg tags (fork choice is Nakamoto's, so as in nakamoto.py):
- Override with h > 0 (InitStates.m:68-106): the honest chain loses h blocks
  (majority, depth h), on both successors (the next block comes after).
- Wait in an Active fork, honest miner on the attacker's branch (:127-136,
  prob. gamma (1-alpha)): majority, depth h.
- Adopt in an Active fork (tag added 2026-11): the same gamma share, minority,
  depth h (none if gamma = 0).
- Wait in an Active fork or Match, honest miner on the honest branch (:123-126,
  :147-149; prob. (1-gamma)(1-alpha)): the gamma share that followed the
  attacker's h published blocks drops them: minority, depth h (none if gamma = 0).
The depth is exact: h is the number of honest blocks since the fork and the
honest chain loses exactly those.

normalization "main_chain" (default): difficulty = Rs + Rh (the reference's rou).
(normalization "time" applies to the disruption objective only; revenue is unaffected.)
normalization "time": every legal transition mines exactly one block (Adopt,
Override and the Match / Active-Wait race all end with the next block drawn,
alpha / 1 - alpha: InitStates.m:33-39, 68-106, 110-159), so it contributes 1 and
rates are per block interval (RS and PRS have no sub-blocks). Illegal: 0.

PRS' reward_coefficient defaults to alpha as in the reference's run.py.
"""
from __future__ import annotations

from typing import Tuple

from .reorgs import MAJORITY, MINORITY, TaggedModel, TaggedTransitions
from .vendor.blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from .vendor.blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from .vendor.blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .vendor.blockchain_mdps.base.blockchain_model import BlockchainModel


def count_blocks(history: int) -> int:
    """countBlocks.m: number of 1 bits."""
    return bin(history).count("1")


class RewardSplittingModel(TaggedModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int = 30, window: int = 6,
                 fork_window: int = 6, objective: str = "revenue", min_depth: int = 11,
                 weight: str = "count", normalization: str = "main_chain"):
        if normalization not in ("main_chain", "time"):
            raise ValueError("normalization must be main_chain or time")
        self.normalization = normalization
        self.time_rates = normalization == "time" and objective == "disruption"
        window, fork_window = int(window), int(fork_window)
        if window < 2:
            raise ValueError("window must be at least 2")
        self.init_objective(objective, min_depth, weight)
        self.alpha = alpha
        self.gamma = gamma
        self.max_fork = int(max_fork)
        self.window = window
        self.fork_window = fork_window
        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum(
            'Action', ['Illegal', 'Adopt', 'Wait', 'Match'] + [f'Override{k}' for k in range(1, window)])
        super().__init__()

    def _args(self) -> tuple:
        return (self.alpha, self.gamma, self.max_fork, self.window, self.fork_window)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}{self._args()}, {self.objective}, "
                f"{self.min_depth}, {self.weight}, {self.normalization})")

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, self._args() + (self.objective, self.min_depth, self.weight,
                                               self.normalization)

    def get_state_space(self):
        return DefaultValueSpace(
            MultiDimensionalDiscreteSpace((0, self.max_fork), (0, self.max_fork), self.Fork,
                                          (0, 2 ** (self.window - 1) - 1)),
            self.get_final_state())

    def get_action_space(self):
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 1, self.Fork.Irrelevant, 0      # baseStateH, InitStates.m:12

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.Fork.Irrelevant, 0

    # InitStates.m:61-62: the adopt rewards (Rs, Rh); the only RS/PRS difference
    def adopt_rewards(self, shared: int, h: int, a: int) -> Tuple[float, float]:
        """`shared` = historySum + min(a, forkWindow) blocks whose reward is split."""
        return shared * 0.5, shared * 0.5 + h - min(a, self.fork_window)

    def _st(self, a: int, h: int, fork, history: int) -> tuple:
        # st2stnum.m:12-13: no fork bit for a < h
        return a, h, (self.Fork.Irrelevant if a < h else fork), history

    def get_tagged_transitions(self, state, action, check_valid: bool = True) -> TaggedTransitions:
        t = self._tagged_transitions(state, action)
        if self.time_rates:
            for nxt in t.difficulty_contributions:
                t.difficulty_contributions[nxt] = 0.0 if nxt == self.final_state else 1.0
        return t

    def _tagged_transitions(self, state, action) -> TaggedTransitions:
        t = TaggedTransitions()
        if state == self.final_state:
            t.add(self.final_state, probability=1)
            return t
        a, h, fork, history = state
        F, A = self.Fork, self.Action
        alpha, gamma, mx, w = self.alpha, self.gamma, self.max_fork, self.window
        hist_sum = count_blocks(history)

        if action is A.Illegal:
            t.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        elif action is A.Adopt:                      # InitStates.m:33-39, 61-64
            rs, rh = self.adopt_rewards(hist_sum + min(a, self.fork_window), h, a)
            # Adopt from an Active fork: the gamma share on the attacker's h published blocks drops them
            drop = [(MINORITY, h)] if fork is F.Active and gamma > 0 else []
            t.add(self._st(0, 1, F.Irrelevant, 0), probability=1 - alpha, reward=rs,
                  difficulty_contribution=rs + rh, reorgs=drop)
            t.add(self._st(1, 0, F.Irrelevant, 0), probability=alpha, reward=rs,
                  difficulty_contribution=rs + rh, reorgs=drop)

        elif action is A.Wait:                       # InitStates.m:110-137
            if fork is not F.Active and a + 1 <= mx and h + 1 <= mx:
                t.add(self._st(a + 1, h, F.Irrelevant, history), probability=alpha)
                t.add(self._st(a, h + 1, F.Relevant, history), probability=1 - alpha)
            elif fork is F.Active and a >= h and h > 0 and a + 1 <= mx and h + 1 <= mx:
                self._race(t, a, h, history, F.Active)
            else:
                t.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action is A.Match:                      # InitStates.m:140-159
            if fork is F.Relevant and a >= h and h > 0 and a + 1 <= mx and h + 1 <= mx:
                self._race(t, a, h, history, F.Active)
            else:
                t.add(self.final_state, probability=1, reward=self.error_penalty)

        else:                                        # override, margin k = 1..window-1
            margin = int(action)  - int(A.Override1) + 1
            if a >= h + margin and a - (h + margin) + 1 <= mx:   # :68
                reorg = [(MAJORITY, h)]
                if h + margin >= w - 1:              # :69-85, history cleared
                    new_history = 2 ** (w - 1) - 2 ** margin
                    rs = hist_sum + (h + margin - (w - 1)) + margin
                else:                                # :86-105
                    kept_len = (w - 1) - (h + margin)
                    kept = history & (2 ** kept_len - 1)
                    rs = count_blocks(history - kept) + margin
                    new_history = kept * 2 ** (h + margin) + 2 ** (h + margin) - 2 ** margin
                t.add(self._st(a - h - margin + 1, 0, F.Irrelevant, new_history), probability=alpha,
                      reward=rs, difficulty_contribution=rs, reorgs=reorg)
                t.add(self._st(a - h - margin, 1, F.Relevant, new_history), probability=1 - alpha,
                      reward=rs, difficulty_contribution=rs, reorgs=reorg)
            else:
                t.add(self.final_state, probability=1, reward=self.error_penalty)
        return t

    def _race(self, t: TaggedTransitions, a: int, h: int, history: int, active) -> None:
        """Wait in an Active fork / Match: InitStates.m:120-136 = :142-158."""
        F = self.Fork
        t.add(self._st(a + 1, h, active, history), probability=self.alpha)
        t.add(self._st(a, h + 1, F.Relevant, history), probability=(1 - self.gamma) * (1 - self.alpha),
              reorgs=[(MINORITY, h)] if self.gamma > 0 else [])
        full = history * 2 ** h + 2 ** h - 1
        new_history = full % 2 ** (self.window - 1)
        reward = count_blocks(full - new_history)
        t.add(self._st(a - h, 1, F.Relevant, new_history), probability=self.gamma * (1 - self.alpha),
              reward=reward, difficulty_contribution=reward, reorgs=[(MAJORITY, h)])


class ProportionalRewardSplittingModel(RewardSplittingModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int = 30, window: int = 6,
                 fork_window: int = 6, reward_coefficient: float = None, objective: str = "revenue",
                 min_depth: int = 11, weight: str = "count", normalization: str = "main_chain"):
        self.reward_coefficient = alpha if reward_coefficient is None else reward_coefficient
        super().__init__(alpha, gamma, max_fork, window, fork_window, objective, min_depth, weight,
                         normalization)

    def _args(self) -> tuple:
        return super()._args() + (self.reward_coefficient,)

    def __reduce__(self) -> Tuple[type, tuple]:
        # (alpha, gamma, max_fork, window, fork_window, reward_coefficient, objective, ...)
        return self.__class__, self._args() + (self.objective, self.min_depth, self.weight,
                                               self.normalization)

    # ProportionalRewardSplitting/InitStates.m:61-64
    def adopt_rewards(self, shared: int, h: int, a: int) -> Tuple[float, float]:
        c = self.reward_coefficient
        return shared * c, shared * (1 - c) + h - min(a, self.fork_window)
