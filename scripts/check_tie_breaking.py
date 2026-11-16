"""Oracle runs for the tie-breaking MDPs (run on box90, never on the control box).

  share PROTO L ALPHA [R]      optimal attacker share (Howard) at one alpha
  pt PROTO L LO HI [R]         profitable threshold by bisection on (LO, HI):
                               profitable iff share > alpha + 1e-4
PROTO in utb, udtb, shtb; R = hash regions (shtb only, default 15).
"""
import sys
import time

from pcmdp.howard import solve_average
from pcmdp.tie_breaking import (SmallestHashTieBreakingModel, UDTieBreakingModel,
                                UniformTieBreakingModel)


def build(proto, L, alpha, R):
    if proto == "utb":
        return UniformTieBreakingModel(alpha, L)
    if proto == "udtb":
        return UDTieBreakingModel(alpha, L)
    return SmallestHashTieBreakingModel(alpha, L, R)


def share(proto, L, alpha, R=15):
    t0 = time.time()
    model = build(proto, L, alpha, R)
    _, g, it = solve_average(model)
    print(f"{proto} L={L} R={R} alpha={alpha:.6f} share={g:.8f} excess={g - alpha:+.3e} "
          f"iters={it} states={model.state_space.size} {time.time() - t0:.0f}s", flush=True)
    return g


def main(argv):
    mode, proto, L = argv[0], argv[1], int(argv[2])
    if mode == "share":
        share(proto, L, float(argv[3]), int(argv[4]) if len(argv) > 4 else 15)
        return
    lo, hi = float(argv[3]), float(argv[4])
    R = int(argv[5]) if len(argv) > 5 else 15
    while hi - lo > 2.5e-5:
        mid = (lo + hi) / 2
        if share(proto, L, mid, R) > mid + 1e-4:
            hi = mid
        else:
            lo = mid
    print(f"PT {proto} L={L} R={R}: ({lo:.5f}, {hi:.5f}) mid {(lo + hi) / 2:.5f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
