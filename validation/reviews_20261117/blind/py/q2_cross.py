"""Cross-check pcmdp ports against the exact ratio solve of the reference matrices at common points,
and the true Nakamoto gamma=0.5 curve for Q1."""
from pcmdp.pop import PublishOrPerishModel
from pcmdp.tie_breaking import SmallestHashTieBreakingModel
from pcmdp.nakamoto import NakamotoModel
from pcmdp import howard
print("== PoP port vs reference exact (maxB=12, k=3)", flush=True)
for alpha in (0.27, 0.30, 0.2640, 0.2641):
    pol, g, it = howard.solve_average(PublishOrPerishModel(alpha, max_b=12, k=3))
    print(f"pop alpha={alpha} max_b=12 howard g={g:.12f} g-alpha={g-alpha:+.3e}", flush=True)
print("== SHTB port vs reference exact (max_fork=40, regions=15)", flush=True)
for alpha in (0.05, 0.065, 0.066, 0.07, 0.10):
    pol, g, it = howard.solve_average(SmallestHashTieBreakingModel(alpha, max_fork=40, regions=15))
    print(f"shtb alpha={alpha} r15 mf40 howard g={g:.12f} g-alpha={g-alpha:+.3e}", flush=True)
print("== Nakamoto gamma=0.5 true curve (max_fork 40 and 80)", flush=True)
for mf in (40, 80):
    for alpha in (0.40, 0.41, 0.42, 0.43, 0.44, 0.45):
        pol, g, it = howard.solve_average(NakamotoModel(alpha, 0.5, mf))
        print(f"nak alpha={alpha} gamma=0.5 max_fork={mf} g={g:.6f}", flush=True)
