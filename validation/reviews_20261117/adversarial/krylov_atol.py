# Review check: scipy bicg with atol=1e-8 on (I-P)V=r when ||r|| is tiny: does it return x0 with info 0?
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl
n, H = 2000, 1e5
rng = np.random.default_rng(0)
# unichain: state i -> i+1 w.p. 0.5, -> 0 w.p. 0.5, scaled by (1-1/H) (pto transformation)
rows = np.r_[np.arange(n), np.arange(n)]; cols = np.r_[(np.arange(n)+1) % n, np.zeros(n, int)]
P = sp.csr_matrix((np.r_[np.full(n, .5), np.full(n, .5)]*(1-1/H), (rows, cols)), shape=(n, n))
A = sp.eye(n) - P
for scale in [1.0, 1e-6, 1e-8, 1e-9]:
    r = np.zeros(n); r[n-1] = scale  # reward on a rare transition
    exact = spl.spsolve(A.tocsc(), r)
    V, info = spl.bicg(A, r, atol=1e-3*1e-5, rtol=1e-7, x0=np.zeros(n))
    print(f"scale={scale:.0e} info={info} exact_V0={exact[0]:.4e} bicg_V0={V[0]:.4e} ratio={V[0]/exact[0]:.3e}")
