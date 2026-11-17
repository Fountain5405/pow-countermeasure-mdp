from pcmdp.nakamoto import NakamotoModel
from pcmdp import howard
ref = {(0.40,0.0):0.4876785278,(0.45,0.0):0.6587448120,(0.40,0.5):0.5725021362,(0.45,0.5):0.7589187622,(0.40,1.0):0.6666641235,(0.45,1.0):0.8177566528}
for gamma in (0.0, 0.5, 1.0):
    for alpha in (0.40, 0.45):
        pol, g, it = howard.solve_average(NakamotoModel(alpha, gamma, 30))
        print(f"alpha={alpha} gamma={gamma} max_fork=30 howard g={g:.10f}  aumayr_lower_bound={ref[(alpha,gamma)]:.10f}  diff={g-ref[(alpha,gamma)]:+.2e}", flush=True)
