"""Fruitchains incentive-compatibility MDP (Zhang & Preneel 2019, App. C).

Port of Aumayr et al.'s MATLAB (GPL-3.0), proportional-reward-splitting-MDP,
incentive_compatibility/Fruitchains/{Init.m, InitStates.m, InitStnum.m,
st2stnum.m, stnum2st.m, SolveMDP.m}. Every transition is one mined unit:
honest block (prob. (1-alpha)/(1+b)), honest fruit ((1-alpha) b/(1+b)),
attacker block (alpha/(1+b)) or attacker fruit (alpha b/(1+b)), b = block_to_fruit.

State (ab, hb, afA, hf, last): attacker's private chain length ab, honest
public chain length hb, attacker fruits mined after the chain reached `window`
(afA; 0 while ab < window, because those fruits are credited at once), honest
fruits not yet credited hf, last = 1 if the last unit was an honest block.
Actions: Adopt (give up the fork, all honest fruits are credited to the honest
side), Override (publish everything; possible if ab >= hb + 1, or ab >= hb when
gamma = 1 and the last unit was an honest block), Wait.

Rewards: Rs = attacker fruits credited, Rh = honest fruits credited; the
transition's reward is Rs and its difficulty contribution Rs + Rh, so the
solved ratio is the reference's rou = Rs / (Rs + Rh) (InitStates.m's
Rs/Rh tables; SolveMDP.m bisects on (1-rou) Rs - rou Rh). Illegal actions go
to pto's final state with the error penalty (the reference sends them to a base
state with Rh = 10000; both are never chosen).

This MDP favours honest miners (the paper: the incentive-compatibility value
is an upper bound on what the attacker can gain from deviating), so rou is
not a tight lower bound on the attacker's best revenue.

Reorg tags. Fruitchains keeps longest-chain fork choice for blocks and the
attacker only ever publishes by Override, so the only public block orphaning
is an override: the hb honest public blocks are removed from the chain most
honest miners follow (majority, depth hb; the four mining outcomes of the
transition all carry it). With gamma = 1 and a tie the honest miners all
switch (gamma is 0 or 1 here), so there is no minority branch and no
Match/race: no (MINORITY, .) tags. The depth is exactly hb, fixed by the
state. Fruits orphaned are not block reorgs and are not tagged. Adopt drops
only private blocks. The cap on hb (max_fork) bounds reported depths.

normalization "fruits" (the reference ratio): difficulty = credited fruits, so
the disruption objective is per credited fruit; "events": every non-illegal
transition counts 1, so rates are per mined unit; "time" (disruption objective only,
revenue is unaffected): every non-illegal
transition is one unit, a block with probability 1/(1+b) (InitStates.m:15-18,
b = bfRatio), so units arrive (1+b) times as fast as blocks and a unit lasts
1/(1+b) block intervals; rates are per block interval.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from .reorgs import MAJORITY, TaggedModel, TaggedTransitions
from .vendor.blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from .vendor.blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from .vendor.blockchain_mdps.base.base_space.space import Space
from .vendor.blockchain_mdps.base.blockchain_model import BlockchainModel


class _PackedSpace(Space):
    """The reference's packing (st2stnum.m, stnum2st.m, 0-based): afA only for ab >= window."""

    def __init__(self, max_fork: int, window: int, max_afa: int, max_hf: int):
        self.mfl, self.win, self.maxafa, self.maxhf = max_fork, window, max_afa, max_hf
        super().__init__()

    def _calc_dimension(self) -> int:
        return 5

    def _calc_size(self) -> int:
        groups = self.win - 1 + (self.mfl - self.win) * (1 + self.maxafa) + self.maxafa + 1 + 1
        return groups * (self.mfl + 1) * (self.maxhf + 1) * 2  # InitStnum.m:5-6

    def element_to_index(self, element: tuple) -> int:
        ab, hb, afa, hf, last = element
        if not (0 <= ab <= self.mfl and 0 <= hb <= self.mfl and 0 <= afa <= self.maxafa
                and 0 <= hf <= self.maxhf and last in (0, 1)) or (ab < self.win and afa):
            raise ValueError("Bad value", element)
        group = ab if ab < self.win else self.win - 1 + (ab - self.win) * (self.maxafa + 1) + afa + 1
        return ((group * (self.mfl + 1) + hb) * (self.maxhf + 1) + hf) * 2 + last

    def index_to_element(self, index: int) -> tuple:
        if index < 0 or index >= self.size:
            raise ValueError("Bad index")
        last = index % 2
        index //= 2
        hf = index % (self.maxhf + 1)
        index //= self.maxhf + 1
        hb = index % (self.mfl + 1)
        group = index // (self.mfl + 1)
        if group < self.win:
            ab, afa = group, 0
        else:
            group -= self.win
            afa = group % (self.maxafa + 1)
            ab = group // (self.maxafa + 1) + self.win
        return ab, hb, afa, hf, last

    def transform_element(self, element: Any) -> Any:
        return element

    def enumerate_dimension(self, coordinate: int) -> List[int]:
        top = (self.mfl, self.mfl, self.maxafa, self.maxhf, 1)[coordinate]
        return list(range(top + 1))


class FruitchainsModel(TaggedModel, BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int = 30, window: int = 6,
                 block_to_fruit: float = 1.0, max_afa: int = 40, max_hf: int = 50,
                 objective: str = "revenue", min_depth: int = 11, weight: str = "count",
                 normalization: str = "fruits"):
        """Init.m's (alpha, mFL, window, bToF, gamma) plus maxafa, maxhf (Init.m:16-17).

        gamma must be 0 or 1 (the reference's tie rule); window <= max_fork.
        """
        if gamma not in (0, 1):
            raise ValueError("gamma must be 0 or 1")
        if not 1 <= window <= max_fork:
            raise ValueError("need 1 <= window <= max_fork")
        if normalization not in ("fruits", "events", "time"):
            raise ValueError("normalization must be fruits, events or time")
        self.alpha, self.gamma = alpha, gamma
        self.max_fork, self.window = max_fork, window
        self.block_to_fruit, self.max_afa, self.max_hf = block_to_fruit, max_afa, max_hf
        self.normalization = normalization
        self.time_rates = normalization == "time" and objective == "disruption"
        self.init_objective(objective, min_depth, weight)
        self.Action = self.create_int_enum("Action", ["Illegal", "Adopt", "Override", "Wait"])
        super().__init__()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.alpha}, {self.gamma}, {self.max_fork}, {self.window}, "
                f"{self.block_to_fruit}, {self.max_afa}, {self.max_hf}, {self.objective}, "
                f"{self.min_depth}, {self.weight}, {self.normalization})")

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.max_fork, self.window, self.block_to_fruit,
                                self.max_afa, self.max_hf, self.objective, self.min_depth, self.weight,
                                self.normalization)

    def get_state_space(self) -> Space:
        return DefaultValueSpace(_PackedSpace(self.max_fork, self.window, self.max_afa, self.max_hf),
                                 self.get_final_state())

    def get_action_space(self) -> Space:
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, 0, 0, 0  # InitStates.m:14 baseStateAf

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, -1, -1, 0

    def get_tagged_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                               check_valid: bool = True) -> TaggedTransitions:
        t = TaggedTransitions()
        if state == self.final_state:
            t.add(self.final_state, probability=1)
            return t
        ab, hb, afa, hf, last = state
        b = self.block_to_fruit
        p_hb = (1 - self.alpha) / (1 + b)          # InitStates.m:15-18
        p_hf = (1 - self.alpha) - p_hb
        p_ab = self.alpha / (1 + b)
        p_af = self.alpha - p_ab
        base_hb, base_hf, base_ab, base_af = (0, 1, 0, 0, 1), (0, 0, 0, 1, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)
        A = self.Action

        def add(nxt, p, rs=0.0, rh=0.0, reorgs=()):
            t.add(nxt, probability=p, reward=rs, difficulty_contribution=rs + rh, reorgs=reorgs)

        if action is A.Illegal:
            t.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        elif action is A.Adopt:  # InitStates.m:43-56, 75, 80: all honest fruits credited
            add(base_hb, p_hb, 0, hf)
            add(base_hf, p_hf, 0, hf)
            add(base_ab, p_ab, 0, hf)
            add(base_af, p_af, 1, hf)

        elif action is A.Override:  # InitStates.m:85-123
            published = hb if (self.gamma == 1 and last == 1) else hb + 1
            if ab >= published:
                rh = 0 if ab >= self.window else hf
                tag = [(MAJORITY, hb)]  # the hb honest public blocks are orphaned
                add(base_hb, p_hb, afa, rh, tag)
                add(base_hf, p_hf, afa, rh, tag)
                add(base_ab, p_ab, afa, rh, tag)
                add(base_af, p_af, afa + 1, rh, tag)
            else:
                t.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action is A.Wait:  # InitStates.m:125-153
            if max(ab + 1, hb + 1) <= self.max_fork and afa < self.max_afa and hf < self.max_hf:
                add((ab, hb + 1, afa, hf, 1), p_hb)
                add((ab, hb, afa, hf + 1, 0), p_hf)
                add((ab + 1, hb, afa, hf, 0), p_ab)
                if ab < self.window:
                    add((ab, hb, afa, hf, 0), p_af, 1)
                else:
                    add((ab, hb, afa + 1, hf, 0), p_af)
            else:
                t.add(self.final_state, probability=1, reward=self.error_penalty)

        if self.normalization == "events":
            for nxt in t.difficulty_contributions:
                t.difficulty_contributions[nxt] = 0.0 if nxt == self.final_state else 1.0
        elif self.time_rates:
            for nxt in t.difficulty_contributions:
                t.difficulty_contributions[nxt] = 0.0 if nxt == self.final_state else 1.0 / (1 + b)
        return t

    def get_honest_revenue(self) -> float:
        return self.alpha if self.objective == "revenue" else 0.0

    def build_honest_policy(self) -> BlockchainModel.Policy:
        """Override whenever possible, else adopt when behind, else wait (adopt at the hf cap)."""
        A = self.Action
        policy = np.zeros(self.state_space.size, dtype=int)
        for i in range(self.state_space.size):
            ab, hb, afa, hf, last = self.state_space.index_to_element(i)
            if ab < 0:
                continue
            published = hb if (self.gamma == 1 and last == 1) else hb + 1
            if ab >= published and ab > 0:
                policy[i] = A.Override
            elif hb > 0 or hf >= self.max_hf or afa >= self.max_afa or max(ab, hb) + 1 > self.max_fork:
                policy[i] = A.Adopt
            else:
                policy[i] = A.Wait
        return tuple(policy)
