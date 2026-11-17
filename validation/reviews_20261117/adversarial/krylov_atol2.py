# Review check: pto-style system (I - (1-1/H) P) V = r, unichain with a rare rewarded transition.
# Compare scipy bicg at mdptoolbox-roi tolerances (atol=1e-3*eps, rtol=1e-7, x0=0) against a direct solve.
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
H = 1e5
for n in [20, 26, 30]:
    rows = np.r_[np.arange(n), np.arange(n)]; cols = np.r_[np.minimum(np.arange(n)+1, n-1), np.zeros(n, int)]
    P = sp.csr_matrix((np.r_[np.full(n, .5), np.full(n, .5)]*(1-1/H), (rows, cols)), shape=(n, n))
    A = (sp.eye(n) - P).tocsc()
    r = np.zeros(n); r[n-1] = 0.5   # reward 1 on the n-1 -> 0 transition (prob .5): expected reward per visit
    exact = spl.spsolve(A, r)
    for fn in [spl.bicg, spl.bicgstab, spl.qmr, spl.gmres]:
        try:
            V, info = fn(A, r, atol=1e-3*1e-5, rtol=1e-7, x0=np.zeros(n))
        except TypeError:
            V, info = fn(A, r, atol=1e-3*1e-5, tol=1e-7, x0=np.zeros(n))
        print(f"n={n} {fn.__name__:9s} info={info} exact_rate={exact[0]/H:.4e} krylov_rate={V[0]/H:.4e} resid={np.linalg.norm(A@V-r):.2e} normb={np.linalg.norm(r):.2e}")
        if info == 0: break
