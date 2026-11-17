"""Independent solver check: exact stationary evaluation + Bellman-residual certificate
for policies returned by howard.solve_average and PTOSolver on pcmdp models."""
import sys, time, json
import numpy as np
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import splu, spsolve
from scipy.sparse.csgraph import breadth_first_order
from pcmdp.nakamoto import NakamotoModel
from pcmdp.sop_walk import SoPWalkModel
from pcmdp import howard
from pcmdp.vendor.blockchain_mdps.base.solver.pto_solver import PTOSolver
from pcmdp.vendor.blockchain_mdps.base.blockchain_mdps.sparse_blockchain_mdp import SparseBlockchainMDP

def tables(model):
    mdp = SparseBlockchainMDP(model); mdp.build_mdp()
    na = mdp.num_of_actions
    P = [csr_matrix(mdp.P.get_data()[a]) for a in range(na)]
    r = np.array([P[a].multiply(csr_matrix(mdp.R.get_data()[a])).sum(axis=1).A.ravel() for a in range(na)])
    d = np.array([P[a].multiply(csr_matrix(mdp.D.get_data()[a])).sum(axis=1).A.ravel() for a in range(na)])
    fin = mdp.final_state_index
    legal = np.array([(P[a][:, fin].toarray().ravel() == 0) & (np.abs(P[a].sum(axis=1).A.ravel() - 1) < 1e-12) for a in range(na)])
    return mdp, P, r, d, legal

def exact_eval(mdp, P, r, d, legal, policy):
    """g of a policy from its stationary distribution (own sparse direct solve) + bias h on all
    non-final states + Bellman residual over legal actions. Returns dict."""
    policy = np.asarray(policy); n = len(policy); na = len(P)
    rows = np.arange(n)
    if not legal[policy, rows][np.arange(n) != mdp.final_state_index].all():
        bad = int((~legal[policy, rows]).sum())
    else:
        bad = 0
    Ppi = sum(diags((policy == a).astype(float)) @ P[a] for a in range(na)).tocsr()
    rpi = r[policy, rows]; dpi = d[policy, rows]
    s0 = mdp.initial_state_index
    reach = np.sort(breadth_first_order(Ppi, s0, directed=True, return_predecessors=False))
    if mdp.final_state_index in set(reach.tolist()):
        return {"illegal_reached": True, "bad_actions": bad}
    sub = Ppi[reach][:, reach]; m = len(reach)
    A = (sub.T - identity(m, format="csr")).tolil()
    j = int(np.searchsorted(reach, s0)); A[j, :] = 1.0
    b = np.zeros(m); b[j] = 1.0
    pi = spsolve(A.tocsc(), b)
    resid = np.abs(pi @ sub - pi).max()
    g = float(pi @ rpi[reach]) / float(pi @ dpi[reach])
    # bias on all non-final states: (I-Ppi) h + g dpi = rpi, h(s0)=0
    keep = np.array([i for i in range(n) if i != mdp.final_state_index])
    Pk = Ppi[keep][:, keep]; k = len(keep)
    B = (identity(k, format="csr") - Pk).tolil()
    js = int(np.searchsorted(keep, s0)); B[:, js] = dpi[keep].reshape(-1, 1)
    z = splu(B.tocsc()).solve(rpi[keep]); g2 = float(z[js]); h = z.copy(); h[js] = 0.0
    hfull = np.zeros(n); hfull[keep] = h
    # Bellman residual over legal actions on reachable states (and all states)
    Q = np.full((na, n), -np.inf)
    for a in range(na):
        q = r[a] - g * d[a] + P[a] @ hfull
        Q[a] = np.where(legal[a], q, -np.inf)
    best = Q.max(axis=0); cur = Q[policy, rows]
    gap = best - cur
    gap_reach = float(gap[reach].max()); gap_all = float(np.nanmax(np.where(np.isfinite(gap), gap, -np.inf)[keep]))
    return {"g_stationary": g, "g_bias_solve": g2, "pi_resid": float(resid), "pi_min": float(pi.min()),
            "n_reach": int(m), "bellman_gap_reach": gap_reach, "bellman_gap_all": gap_all, "bad_actions": bad,
            "states_improvable_reach": int((gap[reach] > 1e-9 * max(1.0, abs(g))).sum())}

def run(model, label, horizons=(10**5,), pto_eps=(1e-5,)):
    t0 = time.time()
    mdp, P, r, d, legal = tables(model)
    out = {"label": label, "n_states": mdp.num_of_states}
    t1 = time.time(); pol_h, g_h, it_h = howard.solve_average(model); t_h = time.time() - t1
    ev = exact_eval(mdp, P, r, d, legal, pol_h)
    out["howard"] = {"g": g_h, "iters": it_h, "secs": round(t_h, 1), **ev}
    for H in horizons:
        for eps in pto_eps:
            t1 = time.time()
            solver = PTOSolver(model, expected_horizon=H)
            pol_p, v_p, it_p, V = solver.calc_opt_policy(epsilon=eps)
            t_p = time.time() - t1
            evp = exact_eval(mdp, P, r, d, legal, pol_p)
            diff = int((np.asarray(pol_p) != np.asarray(pol_h)).sum())
            out[f"pto_H{H:g}_eps{eps:g}"] = {"value": v_p, "iters": it_p, "secs": round(t_p, 1),
                                               "policy_diff_vs_howard": diff, **evp}
    print(json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "revenue":
        for mf in (20, 40):
            for alpha in (0.20, 0.30, 0.35, 0.40, 0.45):
                run(NakamotoModel(alpha, 0.0, mf), f"nak_rev_a{alpha}_g0_mf{mf}", horizons=(10**5, 10**6), pto_eps=(1e-5,))
        for alpha in (0.40, 0.45):
            run(NakamotoModel(alpha, 0.5, 40), f"nak_rev_a{alpha}_g0.5_mf40", horizons=(10**5,), pto_eps=(1e-5,))
    elif which == "disruption":
        for alpha in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            run(NakamotoModel(alpha, 0.0, 40, objective="disruption", min_depth=11, normalization="time"),
                f"nak_dis_a{alpha}_g0_mf40", horizons=(10**5, 10**7), pto_eps=(1e-5, 1e-30))
    elif which == "sop":
        for alpha in (0.10, 0.15, 0.20, 0.30):
            run(SoPWalkModel(alpha), f"sopwalk_a{alpha}", horizons=(10**5,), pto_eps=(1e-5, 1e-30))
