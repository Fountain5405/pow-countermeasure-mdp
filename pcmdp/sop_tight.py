"""SoP (MRL #146, 2026-10-20 rule) for the disruption objective, with the honest pending shares.

The walk model (sop_walk.py) drops the honest pending shares ph when a race
restarts; the exact game (sop_mdp.py) starts the next race at x = -ph after
give_up and x = pa - ph after publish, and an attacker can time its restarts
by ph. This model keeps ph:

  h   honest blocks since the fork, saturating at min_depth;
  a1  whether the attacker has a block since the fork;
  x   the attacker's object lead D + pa - ph (+1 per attacker object, -1 per
      honest object);
  pa  the attacker's pending shares; ph the honest pending shares.

Reveal carry (2026-11-17): under own_pending="count" a reveal publishes the
k = max(0, pa - x + 1) pending shares it needs; published shares are public
(any miner may include them), so carrying them as a private lead would count
them twice. Only pa' = min(pa, x - 1) stay private. "ignore" publishes none.

Honest shares on the fork point (2026-11-18, `honest_shares=True`): until
either side has a block since the fork, an honest share fits the attacker's
rival first block as well as the honest one, in its slot, so it counts for
the attacker too when the attacker lacks that slot (ph >= pa: pa + 1 with
ph + 1, x unchanged). After giving up, the attacker adopts the honest tip and
its pending shares (x = 0, pa = ph). After publishing, the honest shares on
the old fork point fit no child of the published block and drop (ph = 0).
This is MRL #146's slot rule as
tevador's appendix simulator plays it (UNIQUE_SHARES, `adopt_shares`), and
monerosim's race model's `hunt --honest-shares` (campaign 8, section 6.1).
The default (False) is race-model Model B as built: our daemon attacker,
which sees no gossiped shares. `publish_all_on_reveal` publishes every
pending share at a reveal (x = 0, pa = ph = pa), as tevador's attacker does;
it is there to check this model against his simulator.

A reveal is accepted iff x - pa > 0 (sop_mdp's D > ph). The only departures
from the exact game left:
- the window (no reveal before 32 objects past a branch's first block) is
  ignored: in the attacker's favour;
- pa and ph stop counting at their caps while x keeps counting: acceptance
  then reads a smaller pa (in the attacker's favour) and a restart a smaller
  ph (in its favour), but a reveal carries a smaller pa into the next race
  (against it, by the shares above the cap);
- x is clamped to [x_min, x_max]: honest objects below x_min are ignored (in
  its favour), attacker objects above x_max are lost (against it).

Tables are built with numpy over the full product space (no pto: a few
million states) and solved with howard.solve_tables. Time is counted in block
intervals (1/w per object); the reward is 1 per accepted reveal with
h >= min_depth.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

HOLD, REVEAL, GIVE_UP, PUBLISH = 0, 1, 2, 3
ACTION_NAMES = ["hold", "reveal", "give_up", "publish"]


@dataclass
class TightSoP:
    alpha: float
    w: int = 16
    min_depth: int = 11
    x_min: int = -64
    x_max: int = 64
    p_max: int = 32
    q_max: int = 32
    own_pending: str = "ignore"   # "ignore": accept iff x - pa > 0; "count": x > 0
    # True: the pre-2026-11-17 reveal that carries all pa (kept to reproduce old values)
    legacy_carry: bool = False
    # True: the attacker uses the honest shares on the fork point and restarts level (MRL #146)
    honest_shares: bool = False
    # True: a reveal publishes every pending share (tevador's attacker; for the cross-check)
    publish_all_on_reveal: bool = False

    def __post_init__(self):
        if self.own_pending not in ("ignore", "count"):
            raise ValueError("own_pending must be 'ignore' or 'count'")
        self.dims = (self.min_depth + 1, 2, self.x_max - self.x_min + 1, self.p_max + 1, self.q_max + 1)
        self.n = int(np.prod(self.dims))

    def index(self, h, a1, x, pa, ph):
        H, A1, NX, NP, NQ = self.dims
        return (((h * A1 + a1) * NX + (x - self.x_min)) * NP + pa) * NQ + ph

    def decode(self, idx):
        H, A1, NX, NP, NQ = self.dims
        idx = np.asarray(idx)
        ph = idx % NQ; idx = idx // NQ
        pa = idx % NP; idx = idx // NP
        x = idx % NX + self.x_min; idx = idx // NX
        a1 = idx % A1; h = idx // A1
        return h, a1, x, pa, ph

    def accept(self, x, pa):
        return (x > 0) if self.own_pending == "count" else (x - pa > 0)

    @property
    def start(self) -> int:
        return int(self.index(0, 0, 0, 0, 0))

    def from_exact(self, s) -> int:
        """Index of an exact sop_mdp state (a, D, xa, pa, h, xh, ph)."""
        a, D, xa, pa, h, xh, ph = s
        x = max(self.x_min, min(self.x_max, D + pa - ph))
        return int(self.index(min(h, self.min_depth), 1 if a >= 1 else 0, x,
                              min(pa, self.p_max), min(ph, self.q_max)))

    def tables(self):
        """P (list of csr, one per action), r, d, legal (arrays actions x n)."""
        t0 = time.time()
        n = self.n
        h, a1, x, pa, ph = self.decode(np.arange(n))
        clamp = lambda v: np.clip(v, self.x_min, self.x_max)
        al, pb = self.alpha, 1.0 / self.w
        # an honest share also fits the attacker's first block (fork point, its slot free). With
        # no block since the fork D = 0, so x = pa - ph and "slot free" (ph >= pa) is x <= 0; reading
        # it off x avoids the caps (with pa and ph both capped at 24 every honest share looked free,
        # and the optimal policy exploited it: 0.096 against 0.077 in the exact game, 2026-11-18)
        shared = (self.honest_shares & (h == 0) & (a1 == 0) & (x <= 0)).astype(int)
        # hold: four events per object
        nxt = [
            (self.index(h, a1, clamp(x + 1), np.minimum(pa + 1, self.p_max), ph), al * (1 - pb)),   # attacker share
            (self.index(h, 1, clamp(x + 1), 0, ph), al * pb),                                        # attacker block
            (self.index(h, a1, clamp(x - 1 + shared), np.minimum(pa + shared, self.p_max),
                        np.minimum(ph + 1, self.q_max)), (1 - al) * (1 - pb)),                       # honest share
            (self.index(np.minimum(h + 1, self.min_depth), a1, clamp(x - 1), pa, 0), (1 - al) * pb),  # honest block
        ]
        rows = np.concatenate([np.arange(n)] * 4)
        cols = np.concatenate([c for c, _ in nxt])
        vals = np.concatenate([np.full(n, p) for _, p in nxt])
        P_hold = csr_matrix((vals, (rows, cols)), shape=(n, n))
        P_hold.sum_duplicates()
        zeros = np.zeros(n, dtype=int)
        # 2026-11-17: under "count" a reveal publishes k = max(0, pa - x + 1) shares (x = D + pa - ph),
        # which become public (any miner may include them) and drop out of the next race; only
        # pa' = min(pa, x - 1) stay private. Under "ignore" nothing is published.
        pa_r = pa
        if self.own_pending == "count" and not self.legacy_carry:
            pa_r = np.clip(np.minimum(pa, x - 1), 0, None)
        reveal_to = self.index(zeros, zeros, clamp(pa_r), pa_r, zeros)
        if self.publish_all_on_reveal:   # everything public: both tips hold the same pa shares
            reveal_to = self.index(zeros, zeros, zeros, pa, np.minimum(pa, self.q_max))
        publish_to = self.index(zeros, zeros, clamp(pa - ph), pa, ph)
        if self.honest_shares:
            # honest shares on the old fork point fit no child of the published block: they drop
            publish_to = self.index(zeros, zeros, clamp(pa), pa, zeros)
            # adopt the honest tip and its pending shares: level
            give_up_to = self.index(zeros, zeros, zeros, np.minimum(ph, self.p_max), ph)
        else:
            give_up_to = self.index(zeros, zeros, clamp(-ph), zeros, ph)
        one = lambda to: csr_matrix((np.ones(n), (np.arange(n), to)), shape=(n, n))
        P = [P_hold, one(reveal_to), one(give_up_to), one(publish_to)]
        legal = np.zeros((4, n), bool)
        legal[HOLD] = True
        legal[REVEAL] = (a1 == 1) & (h >= 1) & self.accept(x, pa)
        legal[GIVE_UP] = (h >= 1) | (a1 == 1)
        legal[PUBLISH] = (a1 == 1) & (h == 0)
        r = np.zeros((4, n))
        r[REVEAL] = ((h >= self.min_depth) & legal[REVEAL]).astype(float)
        d = np.zeros((4, n))
        d[HOLD] = pb
        self.build_s = time.time() - t0
        return P, r, d, legal

    def initial_policy(self):
        """Give up far behind, reveal when deep and accepted, publish with no fork, else hold."""
        h, a1, x, pa, ph = self.decode(np.arange(self.n))
        pol = np.full(self.n, HOLD)
        pol[(a1 == 1) & (h == 0)] = PUBLISH
        pol[((h >= 1) | (a1 == 1)) & (x < -10)] = GIVE_UP
        pol[(a1 == 1) & (h >= self.min_depth) & self.accept(x, pa)] = REVEAL
        return pol


def profile(P, r, d, policy, start):
    """Rate (reward per block interval) of a policy from its stationary distribution.

    pi(start) is pinned to 1 and the other balance equations are solved by
    sparse LU (no dense normalisation row: that one filled in past 13 GB at
    1.2 M states, 2026-11-16), then pi is normalised. It factors I - P (start
    row and column dropped), the orientation howard.py factors, and solves the
    transposed system: factoring P^T - I filled SuperLU past its memory at 5 M
    states ("Can't expand MemType", x +-64 / p 64, 2026-11-18), as
    reorgs.stationary_distribution found on Fruitchains.
    """
    from scipy.sparse import diags, identity
    from scipy.sparse.csgraph import breadth_first_order
    from scipy.sparse.linalg import splu

    n = P[0].shape[0]
    rows = np.arange(n)
    P_pi = sum(diags((policy == a).astype(float)) @ P[a] for a in range(len(P))).tocsr()
    reach = np.sort(breadth_first_order(P_pi, start, directed=True, return_predecessors=False))
    sub = P_pi[reach][:, reach].tocsr()
    m = len(reach)
    s0 = int(np.searchsorted(reach, start))
    keep = np.r_[0:s0, s0 + 1:m]
    # sum_i pi_i B_ij = 0 for j != start, B = I - P, pi_start = 1:
    # B[keep, keep]^T pi_keep = -B[start, keep]^T
    B = (identity(m, format="csr") - sub).tocsr()
    rest = splu(B[keep][:, keep].tocsc()).solve(-B[s0][:, keep].toarray().ravel(), trans="T")
    pi = np.zeros(m)
    pi[s0] = 1.0
    pi[keep] = rest
    pi = np.clip(pi, 0.0, None)
    pi /= pi.sum()
    rr, dd = r[policy, rows][reach], d[policy, rows][reach]
    return {"rate": float(pi @ rr / (pi @ dd)), "reachable": int(m),
            "give_up_share": float(pi[policy[reach] == GIVE_UP].sum())}


def exact_rate(model: TightSoP, policy, objects: int, seed: int = 1, window: bool = True) -> dict:
    """Deep-reorg rate of `policy` in the exact SoP game (sop_mdp's rules), by simulation.

    Actions the exact game forbids (a reveal before the window or one the
    network rejects) become hold. window=False drops the window test (the
    only rule the tight model leaves out), for a like-for-like check.
    """
    from pcmdp import sop_rules as M

    if model.w != M.W_SHARES:
        raise ValueError(f"exact_rate needs w = {M.W_SHARES}")
    caps = M.Caps(a_max=10 ** 9, h_max=10 ** 9, d_max=10 ** 9, p_max=10 ** 9, deficit=10 ** 9)
    events = np.random.default_rng(seed).choice(4, size=objects, p=np.array(M.event_probs(model.alpha)))
    s = M.START
    deep = 0
    for e in events:
        is_att, is_blk = M.EVENTS[e]
        s = M.event(s, is_att, is_blk, caps, honest_shares=model.honest_shares)
        a, D, xa, pa, h, xh, ph = s
        if a == 0 and h == 0:
            continue
        act = int(policy[model.from_exact(s)])
        accepted = a >= 1 and (M.past_window(s) or not window) and D + (pa if model.own_pending == "count" else 0) > ph
        if act == REVEAL and not accepted:
            act = HOLD
        if act == PUBLISH and not (a >= 1 and h == 0):
            act = HOLD
        if act == REVEAL:
            deep += h >= model.min_depth
            if model.publish_all_on_reveal:
                s = (0, 0, 0, pa, 0, 0, pa)
            else:
                s = M.do_reveal(s, model.own_pending == "count")[0]
        elif act == PUBLISH:
            s = M.do_publish(s, model.honest_shares)[0]
        elif act == GIVE_UP:
            s = M.do_give_up(s, model.honest_shares)[0]
    block_times = objects / model.w
    return {"objects": objects, "deep": deep, "rate": deep / block_times}


def main(argv=None) -> int:
    import argparse
    import json
    import sys

    from .howard import solve_tables

    p = argparse.ArgumentParser(description="Solve the tight SoP disruption model (one JSON line on stdout).")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--min-depth", type=int, default=11)
    p.add_argument("--x-min", type=int, default=-64)
    p.add_argument("--x-max", type=int, default=64)
    p.add_argument("--p-max", type=int, default=32)
    p.add_argument("--q-max", type=int, default=32)
    p.add_argument("--own-pending", choices=["ignore", "count"], default="ignore")
    p.add_argument("--honest-shares", action="store_true",
                   help="the attacker uses the honest shares on the fork point and restarts level (MRL #146)")
    p.add_argument("--save-policy")
    p.add_argument("--load-policy", help="skip the solve: read the policy from this .npz (profile and exact checks)")
    p.add_argument("--no-profile", action="store_true", help="skip the stationary-distribution check of g")
    p.add_argument("--exact-objects", type=int, default=0, help="also run the policy in the exact game")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)

    m = TightSoP(args.alpha, min_depth=args.min_depth, x_min=args.x_min, x_max=args.x_max,
                 p_max=args.p_max, q_max=args.q_max, own_pending=args.own_pending,
                 honest_shares=args.honest_shares)
    P, r, d, legal = m.tables()
    out = {"model": "sop_tight", **vars(args), "states": m.n, "build_s": round(m.build_s, 1)}
    t0 = time.time()
    import contextlib
    if args.load_policy:
        policy = np.load(args.load_policy)["policy"].astype(int)
        if len(policy) != m.n or not legal[policy, np.arange(m.n)].all():
            raise ValueError("the saved policy does not fit this model")
        out["loaded_policy"] = args.load_policy
    else:
        with contextlib.redirect_stdout(sys.stderr):   # the iteration log; stdout carries the JSON
            policy, g, it = solve_tables(P, r, d, legal, m.start, cur=m.initial_policy(), verbose=True)
        out.update({"g": g, "iterations": it, "solve_s": round(time.time() - t0, 1)})
    if args.save_policy and not args.load_policy:
        np.savez_compressed(args.save_policy, policy=policy.astype(np.int8), args=json.dumps(vars(args)))
    if not args.no_profile:
        t0 = time.time()
        try:
            out["profile"] = profile(P, r, d, policy, m.start)
        except MemoryError as e:   # the check's LU can outgrow the memory cap (alpha 0.30, 2026-11-16)
            out["profile"] = {"error": f"MemoryError: {e}"}
        out["profile_s"] = round(time.time() - t0, 1)
    if args.exact_objects:
        for window in (False, True):
            t0 = time.time()
            res = exact_rate(m, policy, args.exact_objects, seed=args.seed, window=window)
            res["s"] = round(time.time() - t0, 1)
            out["exact_window" if window else "exact_no_window"] = res
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
