from q3_check import run
from pcmdp.nakamoto import NakamotoModel
from pcmdp.sop_walk import SoPWalkModel
for alpha in (0.06, 0.08, 0.10, 0.12):
    run(NakamotoModel(alpha, 0.0, 40, objective="disruption", min_depth=11, normalization="time"),
        f"nak_dis_a{alpha}_g0_mf40", horizons=(10**5, 10**7), pto_eps=(1e-5, 1e-30))
for alpha in (0.10, 0.15, 0.20, 0.30):
    run(SoPWalkModel(alpha), f"sopwalk_a{alpha}", horizons=(10**5,), pto_eps=(1e-5, 1e-30))
