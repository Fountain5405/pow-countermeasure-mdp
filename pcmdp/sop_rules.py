"""Exact transition rules of the SoP game (MRL #146, 2026-10-20 rule, race-model Model B).

Copied verbatim from monerosim (https://github.com/Fountain5405/monerosim) at commit ef3d6453,
studies/sop/scripts/sop_mdp.py, whose module docstring explains the state
(a, D, xa, pa, h, xh, ph) and the rules. Used here to run walk/tight-model
policies in the exact game.
"""
from __future__ import annotations

from typing import List, Tuple

W_SHARES = 16          # objects per block on average (w)
K_OBJECTS = 48         # SoP window in objects (K)
WINDOW = K_OBJECTS - W_SHARES   # either_window: 32 objects past a branch's first block

HOLD, PUBLISH, REVEAL, GIVE_UP = 0, 1, 2, 3
ACTION_NAMES = ["hold", "publish", "reveal", "give_up"]
State = Tuple[int, int, int, int, int, int, int]   # a, D, xa, pa, h, xh, ph
START: State = (0, 0, 0, 0, 0, 0, 0)
EVENTS = [(True, True), (True, False), (False, True), (False, False)]   # (attacker?, block?)
FIELDS = ["a", "D", "xa", "pa", "h", "xh", "ph"]


class Caps:
    def __init__(self, a_max=8, h_max=8, d_max=160, p_max=48, deficit=48):
        self.a_max, self.h_max, self.d_max, self.p_max, self.deficit = a_max, h_max, d_max, p_max, deficit

    def __str__(self):
        return f"a<={self.a_max} h<={self.h_max} |D|<={self.d_max} p<={self.p_max} deficit<={self.deficit}"


def event_probs(alpha: float) -> List[float]:
    pb = 1.0 / W_SHARES
    return [alpha * pb, alpha * (1 - pb), (1 - alpha) * pb, (1 - alpha) * (1 - pb)]


def _collapse(s: State) -> State:
    a, D, xa, pa, h, xh, ph = s
    if max(xa, xh) >= WINDOW:
        xa = xh = WINDOW
    return (a, D, xa, pa, h, xh, ph)


def past_window(s: State) -> bool:
    return max(s[2], s[5]) >= WINDOW


def deficit(s: State) -> int:
    """Honest objects since the fork minus the attacker's."""
    a, D, xa, pa, h, xh, ph = s
    return -D + ph - pa


def accepted(s: State, own_pending: bool) -> bool:
    a, D, xa, pa, h, xh, ph = s
    if a == 0 or not past_window(s):
        return False
    return D + (pa if own_pending else 0) > ph


def event(s: State, is_attacker: bool, is_block: bool, caps: Caps, honest_shares: bool = False) -> State:
    """One object. honest_shares (2026-11-18, MRL #146's slot rule as tevador's simulator plays
    it): before either side's first block since the fork, an honest share also fits the
    attacker's rival first block, in its slot, so it joins the attacker's pending shares when
    the attacker lacks that slot (ph >= pa)."""
    a, D, xa, pa, h, xh, ph = s
    if not is_block:
        if is_attacker:
            pa = min(pa + 1, caps.p_max)
        else:
            if honest_shares and a == 0 and h == 0 and ph >= pa:
                pa = min(pa + 1, caps.p_max)
            ph = min(ph + 1, caps.p_max)
        return (a, D, xa, pa, h, xh, ph)
    if is_attacker:
        if a >= 1:
            xa = min(xa + 1 + pa, WINDOW)
        a, D, pa = a + 1, D + 1 + pa, 0
    else:
        if h >= 1:
            xh = min(xh + 1 + ph, WINDOW)
        h, D, ph = h + 1, D - 1 - ph, 0
    return _collapse((a, D, xa, pa, h, xh, ph))


def do_publish(s: State, honest_shares: bool = False):
    """Publish the attacker's blocks (no honest block since the fork). With honest_shares the
    honest pending shares, mined on the old fork point, fit no child of the published block
    and drop; Model B as built keeps them."""
    a, D, xa, pa, h, xh, ph = s
    return (0, 0, 0, pa, 0, 0, 0 if honest_shares else ph), a, 0


def do_reveal(s: State, own_pending: bool = False):
    """Reveal the attacker chain.

    2026-11-17: under own_pending ("count", MRL #146's text) the attacker
    publishes the k = max(0, ph - D + 1) pending shares it needs for D + k > ph.
    Published shares extend the new tip, are relayed, and can be included by
    any miner, so they are no longer a private lead (carrying them would count
    them twice); only pa - k stay private. Under "ignore" nothing is published.
    """
    a, D, xa, pa, h, xh, ph = s
    k = max(0, ph - D + 1) if own_pending else 0
    return (0, 0, 0, pa - k, 0, 0, 0), a, 0


def do_give_up(s: State, honest_shares: bool = False):
    """Adopt the honest chain. With honest_shares the attacker also adopts the honest pending
    shares on its tip (they are public), so the next race starts level."""
    a, D, xa, pa, h, xh, ph = s
    return (0, 0, 0, ph if honest_shares else 0, 0, 0, ph), 0, h


def at_cap(s: State, caps: Caps) -> bool:
    a, D, xa, pa, h, xh, ph = s
    return (a >= caps.a_max or h >= caps.h_max or abs(D) >= caps.d_max or deficit(s) > caps.deficit)

