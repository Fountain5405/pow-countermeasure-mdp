from pcmdp.tie_breaking import SmallestHashTieBreakingModel
from pcmdp.nakamoto import NakamotoModel
from pcmdp import howard
print("== SHTB port vs reference exact (max_fork=10, regions=15)", flush=True)
for alpha in (0.0575, 0.0580078125, 0.05804687500000001, 0.06, 0.10):
    pol, g, it = howard.solve_average(SmallestHashTieBreakingModel(alpha, max_fork=10, regions=15))
    print(f"shtb alpha={alpha} r15 mf10 howard g={g:.12f} g-alpha={g-alpha:+.3e}", flush=True)
print("== Nakamoto gamma=0.5 true curve (max_fork 40 and 80)", flush=True)
for mf in (40, 80):
    for alpha in (0.40, 0.41, 0.42, 0.43, 0.44, 0.45):
        pol, g, it = howard.solve_average(NakamotoModel(alpha, 0.5, mf))
        print(f"nak alpha={alpha} gamma=0.5 max_fork={mf} g={g:.6f}", flush=True)
