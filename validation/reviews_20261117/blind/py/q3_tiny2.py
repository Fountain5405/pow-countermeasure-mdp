from q3_check import run
from pcmdp.nakamoto import NakamotoModel
for alpha in (0.04, 0.05):
    run(NakamotoModel(alpha, 0.0, 40, objective="disruption", min_depth=11, normalization="time"),
        f"nak_dis_a{alpha}_g0_mf40", horizons=(10**5, 10**7, 10**9), pto_eps=(1e-5, 1e-30))
