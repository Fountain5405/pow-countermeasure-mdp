"""Exact ratio-objective policy iteration (own implementation) on matrices exported from ZP code.
Objective: maximise long-run Rs / (Rs + Rh). P is affine in alpha; two exports give the affine form."""
import sys, json, time
import numpy as np
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import splu, spsolve
from scipy.sparse.csgraph import breadth_first_order

def load_trip(path):
    with open(path, "rb") as f:
        n = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        r = np.fromfile(f, dtype=np.int32, count=n) - 1
        c = np.fromfile(f, dtype=np.int32, count=n) - 1
        v = np.fromfile(f, dtype=np.float64, count=n)
    return r, c, v

def load_model(prefix, a1, a2, nact):
    """returns n, list over actions of (rows, cols, p0, p1, rs, rh) with P = p0 + p1*alpha."""
    acts = []
    n = 0
    for k in range(1, nact + 1):
        r1, c1, v1 = load_trip(f"{prefix}_a{a1:.2f}_P{k}.bin")
        r2, c2, v2 = load_trip(f"{prefix}_a{a2:.2f}_P{k}.bin")
        n = max(n, r1.max() + 1, c1.max() + 1)
        # align by key
        k1 = r1.astype(np.int64) * (1 << 31) + c1; k2 = r2.astype(np.int64) * (1 << 31) + c2
        keys = np.union1d(k1, k2)
        p1v = np.zeros(len(keys)); p2v = np.zeros(len(keys))
        p1v[np.searchsorted(keys, k1)] = v1; p2v[np.searchsorted(keys, k2)] = v2
        slope = (p2v - p1v) / (a2 - a1); icpt = p1v - slope * a1
        rows = (keys >> 31).astype(np.int64); cols = (keys & ((1 << 31) - 1)).astype(np.int64)
        def rew(tag):
            rr, cc, vv = load_trip(f"{prefix}_a{a1:.2f}_{tag}{k}.bin")
            rr2, cc2, vv2 = load_trip(f"{prefix}_a{a2:.2f}_{tag}{k}.bin")
            assert np.array_equal(rr, rr2) and np.array_equal(cc, cc2) and np.array_equal(vv, vv2), "rewards depend on alpha?"
            kk = rr.astype(np.int64) * (1 << 31) + cc
            out = np.zeros(len(keys)); idx = np.searchsorted(keys, kk)
            ok = (idx < len(keys)) & (keys[np.minimum(idx, len(keys) - 1)] == kk)
            out[idx[ok]] = vv[ok]
            return out
        acts.append((rows, cols, icpt, slope, rew("Rs"), rew("Rh")))
    for a in acts:
        n = max(n, a[0].max() + 1, a[1].max() + 1)
    return n, acts

def build(n, acts, alpha):
    P, r, d, legal = [], [], [], []
    for rows, cols, icpt, slope, rs, rh in acts:
        p = icpt + slope * alpha
        assert (p >= -1e-12).all()
        Pa = csr_matrix((p, (rows, cols)), shape=(n, n))
        ra = csr_matrix((p * rs, (rows, cols)), shape=(n, n)).sum(axis=1).A.ravel()
        da = csr_matrix((p * (rs + rh), (rows, cols)), shape=(n, n)).sum(axis=1).A.ravel()
        rowsum = Pa.sum(axis=1).A.ravel()
        # illegal default: P=1 to base with Rh=10000 -> d >= 10000
        leg = (np.abs(rowsum - 1) < 1e-9) & (da < 5000)
        P.append(Pa); r.append(ra); d.append(da); legal.append(leg)
    return P, np.array(r), np.array(d), np.array(legal)

def evaluate(P, r, d, policy, s0):
    n = len(policy); na = len(P); rows = np.arange(n)
    Ppi = sum(diags((policy == a).astype(float)) @ P[a] for a in range(na)).tocsr()
    rpi = r[policy, rows]; dpi = d[policy, rows]
    A = (identity(n, format="csr") - Ppi).tolil()
    A[:, s0] = dpi.reshape(-1, 1)
    z = splu(A.tocsc()).solve(rpi)
    g = float(z[s0]); h = z.copy(); h[s0] = 0.0
    return g, h, Ppi, rpi, dpi

def howard(P, r, d, legal, s0, verbose=False, max_iter=100):
    na, n = r.shape; rows = np.arange(n)
    policy = np.zeros(n, dtype=int)  # action 1 (adopt) is legal everywhere
    assert legal[0].all()
    for it in range(1, max_iter + 1):
        g, h, Ppi, rpi, dpi = evaluate(P, r, d, policy, s0)
        Q = np.full((na, n), -np.inf)
        for a in range(na):
            Q[a] = np.where(legal[a], r[a] - g * d[a] + P[a] @ h, -np.inf)
        best = Q.max(axis=0); cur = Q[policy, rows]
        improve = best > cur + 1e-11
        if verbose: print(f"   it {it} g={g:.12f} changes={improve.sum()}", flush=True)
        if not improve.any():
            break
        policy = np.where(improve, Q.argmax(axis=0), policy)
    # stationary check on reachable states
    reach = np.sort(breadth_first_order(Ppi, s0, directed=True, return_predecessors=False))
    sub = Ppi[reach][:, reach]; m = len(reach)
    B = (sub.T - identity(m, format="csr")).tolil(); j = int(np.searchsorted(reach, s0)); B[j, :] = 1.0
    b = np.zeros(m); b[j] = 1.0
    pi = spsolve(B.tocsc(), b)
    g_stat = float(pi @ rpi[reach]) / float(pi @ dpi[reach])
    return g, g_stat, policy, it, m, float(pi.min())

if __name__ == "__main__":
    prefix, nact, s0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    mode = sys.argv[4]
    n, acts = load_model(prefix, 0.30, 0.40, nact)
    print(f"loaded {prefix}: n={n}", flush=True)
    def solve(alpha):
        t = time.time()
        P, r, d, legal = build(n, acts, alpha)
        g, g_stat, pol, it, m, pimin = howard(P, r, d, legal, s0)
        print(json.dumps({"prefix": prefix, "alpha": alpha, "g": g, "g_stat": g_stat, "g_minus_alpha": g - alpha,
                          "iters": it, "reach": m, "pi_min": pimin, "secs": round(time.time() - t, 1)}), flush=True)
        return g
    if mode == "grid":
        for a in map(float, sys.argv[5:]):
            solve(a)
    elif mode == "bisect":
        lo, hi, tol = float(sys.argv[5]), float(sys.argv[6]), 1e-9
        glo = solve(lo); ghi = solve(hi)
        print(f"bracket lo={lo} g-a={glo-lo:.3e}  hi={hi} g-a={ghi-hi:.3e}", flush=True)
        while hi - lo > 1e-5:
            mid = 0.5 * (lo + hi); g = solve(mid)
            if g > mid + tol: hi = mid
            else: lo = mid
        print(f"PT in ({lo:.6f}, {hi:.6f}]", flush=True)
