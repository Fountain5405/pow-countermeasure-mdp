"""Step the exact game and the tight model through the same events; print the first divergences.

After every event and action, the exact state mapped into the model (from_exact)
must be a successor of the previous model state under the same action, with the
event's probability. Prints up to 10 mismatches.
"""
import numpy as np

from pcmdp import sop_rules as M
from pcmdp.sop_tight import GIVE_UP, HOLD, PUBLISH, REVEAL, TightSoP

m = TightSoP(alpha=0.45, min_depth=3, x_min=-40, x_max=40, p_max=24, q_max=24, own_pending="count",
             honest_shares=True)
P, r, d, legal = m.tables()
pol = m.initial_policy()
caps = M.Caps(a_max=10 ** 9, h_max=10 ** 9, d_max=10 ** 9, p_max=10 ** 9, deficit=10 ** 9)
probs = M.event_probs(m.alpha)
rng = np.random.default_rng(5)
s = M.START
t = m.from_exact(s)
bad = 0
for step in range(200000):
    e = rng.choice(4, p=probs)
    is_att, is_blk = M.EVENTS[e]
    s2 = M.event(s, is_att, is_blk, caps, honest_shares=True)
    t2 = m.from_exact(s2)
    row = P[HOLD][t]
    succ = dict(zip(row.indices.tolist(), row.data.tolist()))
    if t2 not in succ or abs(succ[t2] - probs[e]) > 1e-12 and not any(
            abs(succ[t2] - p) < 1e-12 for p in probs):
        bad += 1
        print(f"HOLD mismatch at {step}: exact {s} --{['att blk', 'att shr', 'hon blk', 'hon shr'][e]}--> {s2}; "
              f"model {tuple(int(v) for v in m.decode(t))} -> expected {tuple(int(v) for v in m.decode(t2))}, "
              f"model successors {[tuple(int(v) for v in m.decode(j)) for j in succ]}")
    s, t = s2, t2
    a, D, xa, pa, h, xh, ph = s
    if a == 0 and h == 0:
        continue
    act = int(pol[t])
    if act == REVEAL and not (a >= 1 and D + pa > ph):
        act = HOLD
    if act == PUBLISH and not (a >= 1 and h == 0):
        act = HOLD
    if act == HOLD:
        continue
    s2 = {REVEAL: lambda: M.do_reveal(s, True)[0], PUBLISH: lambda: M.do_publish(s, True)[0],
          GIVE_UP: lambda: M.do_give_up(s, True)[0]}[act]()
    t2 = m.from_exact(s2)
    j = int(P[act][t].indices[0])
    if j != t2:
        bad += 1
        print(f"{['hold', 'reveal', 'give_up', 'publish'][act]} mismatch at {step}: exact {s} -> {s2}; "
              f"model {tuple(int(v) for v in m.decode(t))} -> {tuple(int(v) for v in m.decode(j))}, "
              f"expected {tuple(int(v) for v in m.decode(t2))}")
    s, t = s2, t2
    if bad >= 10:
        break
print("mismatches:", bad)
