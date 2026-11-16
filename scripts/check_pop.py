"""Oracle runs for the PoP model (run on box90): python scripts/check_pop.py share|pt|nak [max_b ...]"""
import sys
import time

from pcmdp.howard import solve_average
from pcmdp.nakamoto import NakamotoModel
from pcmdp.pop import PublishOrPerishModel


def share(alpha, max_b, **kw):
    t0 = time.time()
    m = PublishOrPerishModel(alpha, max_b, **kw)
    _, g, it = solve_average(m)
    print(f"alpha={alpha} max_b={max_b} {kw} share={g:.6f} it={it} states={m.state_space.size} "
          f"{time.time() - t0:.0f}s", flush=True)
    return g


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "share":
        for b in sys.argv[2:]:
            share(0.3901, int(b))
    elif mode == "nak":
        _, g, _ = solve_average(NakamotoModel(0.3901, 0.0, 80))
        print(f"nakamoto 0.3901 gamma 0: {g:.6f}", flush=True)
    elif mode == "pt":
        max_b = int(sys.argv[2])
        lo, hi = 0.25, 0.275          # Threshold.m: [0.25, 0.275], eps 1e-4
        while hi - lo > 1e-4 / 2:
            a = (lo + hi) / 2
            g = share(a, max_b)
            if g >= a + 1e-4:
                hi = a
            else:
                lo = a
            print(f"  -> [{lo:.6f}, {hi:.6f}]", flush=True)
        print(f"PT in [{lo:.5f}, {hi:.5f}]")
