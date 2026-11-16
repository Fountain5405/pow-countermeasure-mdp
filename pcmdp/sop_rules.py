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


def event(s: State, is_attacker: bool, is_block: bool, caps: Caps) -> State:
    a, D, xa, pa, h, xh, ph = s
    if not is_block:
        if is_attacker:
            pa = min(pa + 1, caps.p_max)
        else:
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


def do_publish(s: State):
    a, D, xa, pa, h, xh, ph = s
    return (0, 0, 0, pa, 0, 0, ph), a, 0


def do_reveal(s: State):
    a, D, xa, pa, h, xh, ph = s
    return (0, 0, 0, pa, 0, 0, 0), a, 0


def do_give_up(s: State):
    a, D, xa, pa, h, xh, ph = s
    return (0, 0, 0, 0, 0, 0, ph), 0, h


def at_cap(s: State, caps: Caps) -> bool:
    a, D, xa, pa, h, xh, ph = s
    return (a >= caps.a_max or h >= caps.h_max or abs(D) >= caps.d_max or deficit(s) > caps.deficit)

