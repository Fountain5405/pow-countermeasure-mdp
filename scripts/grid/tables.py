"""Tables and a jsonl digest for the 2026-11-17 countermeasure grid.

    python scripts/grid/tables.py <out-dir> --jsonl grid.jsonl --md TABLES.md \
        [--second-stage <ss-dir> --ss-jsonl second_stage.jsonl] [--sop-dir <sop-dir>]

<out-dir> holds one <name>.json per solve, as written by run1.sh for the jobs
of gen_20261117.sh: <protocol>_a<alpha>[_g<gamma>]_<rev|dc|dd>, and for SoP
soptight_a<alpha>_dc[_count|_count64]. The jsonl has one record per job: its
scalars, the profile scalars and the reorg-depth rates to 6 significant
figures. The markdown has the cross-protocol tables:

- revenue: the attacker's share of main-chain blocks under the
  revenue-optimal policy, and the first grid alpha where it beats honest;
- disruption: reorgs of 11+ blocks per block interval under the policy that
  maximises them, with the mean time between them at Monero's 120 s blocks,
  and the attacker's share while doing so;
- with --second-stage, the fewest and the most deep reorgs per block interval
  over the revenue-optimal policies (pcmdp.second_stage);
- the mean depth of the 11+ reorgs under both disruption objectives.

Values are printed as solved. Depth rates of order 1e-15 at depths the
policy never reorgs are rounding noise of the stationary solve.
"""

import argparse
import json
import math
import re
from pathlib import Path

GAMMA_PROTOCOLS = ["nakamoto", "rs", "prs", "subchains", "fruitchains"]
TIE_PROTOCOLS = ["utb", "udtb", "shtb", "pop"]
LABEL = {
    "nakamoto": "Nakamoto", "rs": "RS", "prs": "PRS", "subchains": "Subchains",
    "fruitchains": "Fruitchains", "utb": "UTB", "udtb": "UDTB", "shtb": "SHTB",
    "pop": "PoP", "sop": "SoP (#146)", "sop_modelb": "SoP, daemon attacker",
    "sop_ignore": "SoP, patched daemon",
}
# SoP variants (pcmdp.sop_tight): "sop" = MRL #146 (--own-pending count --honest-shares, from
# --sop-dir); "sop_modelb" = count without honest shares (race-model Model B, our daemon's
# attacker, `_count64`); "sop_ignore" = the patched daemon's acceptance rule (`_dc`, caps
# x +-40 / pending 24); `_count` (pending 24, pre-08fb4b4 carry) is superseded and left out.
SOP_NAMES = {None: "sop_ignore", "count64": "sop_modelb", "count": "sop_count_superseded"}
NAME = re.compile(r"^(?P<p>[a-z]+)_a(?P<a>[0-9.]+)(?:_g(?P<g>[0-9.]+))?_(?P<o>rev|dc|dd)(?:_(?P<v>count|count64))?$")
BLOCK_S = 120.0


def sig(x, n=6):
    if isinstance(x, float) and x != 0 and math.isfinite(x):
        return float(f"{x:.{n}g}")
    return x


def load(out_dir):
    recs = []
    for f in sorted(Path(out_dir).glob("*.json")):
        m = NAME.match(f.stem)
        if not m:
            continue
        r = json.loads(f.read_text())
        rec = {"name": f.stem, "protocol": m["p"], "alpha": float(m["a"]),
               "gamma": float(m["g"]) if m["g"] is not None else None,
               "objective": m["o"], "variant": m["v"]}
        if m["p"] == "soptight":
            rec["protocol"] = SOP_NAMES[m["v"]]
            rec["value"] = r["g"]
        for k, v in r.items():
            if k == "profile":
                for pk, pv in v.items():
                    if isinstance(pv, dict):
                        rec[f"profile.{pk}"] = {d: sig(x) for d, x in pv.items()}
                    else:
                        rec[f"profile.{pk}"] = sig(pv)
            elif k not in ("g",) and k not in rec:
                rec[k] = sig(v) if isinstance(v, float) else v
        recs.append(rec)
    return recs


def every(x):
    """Mean time between events at rate x per 120 s block interval."""
    if not x or x <= 0:
        return "never"
    s = BLOCK_S / x
    for unit, size in (("y", 365.25 * 86400), ("d", 86400), ("h", 3600), ("min", 60)):
        if s >= size:
            v = s / size
            if v >= 1e4:
                return f"{v:.0e} {unit}"
            return f"{v:.2g} {unit}" if v < 10 else f"{v:,.0f} {unit}"
    return f"{s:.0f} s"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def load_sop(sop_dir):
    """MRL #146 SoP runs (sop_tight --own-pending count --honest-shares): one record per alpha,
    the largest caps where an alpha was run more than once."""
    best = {}
    for f in sorted(Path(sop_dir).glob("*.json")):
        r = json.loads(f.read_text())
        a = float(r["alpha"])
        size = (r["x_max"] - r["x_min"]) * r["p_max"]
        if a in best and best[a][0] >= size:
            continue
        rec = {"name": f.stem, "protocol": "sop", "alpha": a, "gamma": None, "objective": "dc",
               "variant": "honest_shares", "value": r["g"]}
        for k, v in r.items():
            if k == "profile":
                rec.update({f"profile.{pk}": sig(pv) for pk, pv in v.items()})
            elif isinstance(v, dict):
                rec[k] = {kk: sig(vv) for kk, vv in v.items()}
            elif k != "g":
                rec[k] = sig(v) if isinstance(v, float) else v
        best[a] = (size, rec)
    return [rec for _, rec in best.values()]


def load_second_stage(ss_dir):
    """second_stage.py outputs, keyed by (protocol, alpha, gamma), and one flat record each."""
    ss, flat = {}, []
    for f in sorted(Path(ss_dir).glob("*.json")):
        r = json.loads(f.read_text())
        ss[(r["protocol"], r["alpha"], r["gamma"])] = r
        rec = {"name": f.stem, **{k: (sig(v) if isinstance(v, float) else v) for k, v in r.items()
                                  if not isinstance(v, dict)}}
        for t in ("min", "howard", "max"):
            if "g" in r[t]:
                rec[f"{t}.g"] = sig(r[t]["g"])
            for k, v in r[t]["profile"].items():
                rec[f"{t}.profile.{k}"] = ({d: sig(x) for d, x in v.items()} if isinstance(v, dict)
                                           else sig(v))
        flat.append(rec)
    return ss, flat


def build(recs, ss=None):
    ss = ss or {}
    idx = {(r["protocol"], r["alpha"], r["gamma"], r["objective"]): r
           for r in recs if r["protocol"] != "sop_count_superseded"}
    alphas = sorted({r["alpha"] for r in recs})
    gammas = sorted({r["gamma"] for r in recs if r["gamma"] is not None})
    md = []

    def get(p, a, g, o):
        return idx.get((p, a, g, o))

    # Models and caps: a capped model is a lower bound wherever its cap binds.
    md.append("## Models and caps\n")
    rows = []
    for p in GAMMA_PROTOCOLS + TIE_PROTOCOLS + ["sop", "sop_modelb", "sop_ignore"]:
        rs = [r for r in recs if r["protocol"] == p]
        if not rs:
            continue
        r = rs[0]
        cap = (f"x {r['x_min']}..{r['x_max']}, p {r['p_max']}, q {r['q_max']}"
               if p.startswith("sop") else str(r.get("max_fork")))
        model = "tight SoP model" if p.startswith("sop") else r.get("model", "")
        model = re.sub(r"\((0\.\d+), ", "(alpha, ", model, count=1)
        rows.append([LABEL[p], f"`{model}`", cap, f"{max(x['states'] for x in rs):,}",
                     str(len(rs))])
    md.append(table(["protocol", "model (first job)", "cap", "states", "jobs"], rows) + "\n")

    # Revenue
    md.append("## Revenue: the attacker's share under the revenue-optimal policy\n")
    md.append("A share above alpha is profitable. `=` marks the honest policy "
              "(share = alpha within 1e-6).\n")

    def rev_cell(r):
        if r is None:
            return ""
        v = r["value"]
        return "=" if abs(v - r["honest_value"]) <= 1e-6 else f"{v:.4f}"

    for g in gammas:
        cols = [p for p in GAMMA_PROTOCOLS if any(get(p, a, g, "rev") for a in alphas)]
        md.append(f"### gamma {g:g}\n")
        md.append(table(["alpha"] + [LABEL[p] for p in cols],
                        [[f"{a:.2f}"] + [rev_cell(get(p, a, g, "rev")) for p in cols]
                         for a in alphas]) + "\n")
    md.append("### Protocols with their own tie rule (UTB is gamma 0.5 by construction)\n")
    md.append(table(["alpha"] + [LABEL[p] for p in TIE_PROTOCOLS],
                    [[f"{a:.2f}"] + [rev_cell(get(p, a, 0.5 if p == "utb" else None, "rev"))
                                     for p in TIE_PROTOCOLS] for a in alphas]) + "\n")

    # Thresholds
    md.append("### First grid alpha where the optimum beats honest by more than 1e-4\n")
    rows = []
    for p in GAMMA_PROTOCOLS + TIE_PROTOCOLS:
        cells = []
        for g in gammas if p in GAMMA_PROTOCOLS else [None]:
            gg = 0.5 if p == "utb" else g
            hit = next((a for a in alphas if (r := get(p, a, gg, "rev"))
                        and r["value"] - r["honest_value"] > 1e-4), None)
            have = any(get(p, a, gg, "rev") for a in alphas)
            cells.append("" if not have else (f"{hit:.2f}" if hit else "none"))
        if p in GAMMA_PROTOCOLS:
            rows.append([LABEL[p]] + cells)
        else:
            rows.append([LABEL[p]] + [cells[0] + " (all gamma)"] + [""] * (len(gammas) - 1))
    md.append(table(["protocol"] + [f"gamma {g:g}" for g in gammas], rows) + "\n")

    # Disruption
    md.append("## Disruption: reorgs of 11+ blocks per block interval, disruption-optimal policy\n")
    md.append("Each cell: rate (mean time between them at 120 s blocks; the attacker's "
              "share of main-chain blocks while doing so).\n")

    def dis_cell(r):
        if r is None:
            return ""
        v = r["value"]
        share = r.get("profile.revenue")
        tail = f"; {share:.3f}" if isinstance(share, float) else ""
        return f"{v:.3g} ({every(v)}{tail})"

    for g in gammas:
        cols = [p for p in GAMMA_PROTOCOLS if any(get(p, a, g, "dc") for a in alphas)]
        md.append(f"### gamma {g:g}\n")
        md.append(table(["alpha"] + [LABEL[p] for p in cols],
                        [[f"{a:.2f}"] + [dis_cell(get(p, a, g, "dc")) for p in cols]
                         for a in alphas]) + "\n")
    tie_cols = TIE_PROTOCOLS + [p for p in ("sop", "sop_modelb", "sop_ignore")
                                if any(get(p, a, None, "dc") for a in alphas)]
    md.append("### Protocols with their own tie rule, and SoP\n")
    md.append(table(["alpha"] + [LABEL[p] for p in tie_cols],
                    [[f"{a:.2f}"] + [dis_cell(get(p, a, 0.5 if p == "utb" else None, "dc"))
                                     for p in tie_cols] for a in alphas]) + "\n")

    # Revenue-optimal attacker's deep reorgs: the second stage's range
    if ss:
        md.append("## Deep reorgs (11+) by the revenue-optimal attacker: fewest to most, "
                  "per block interval\n")
        md.append("Over every policy that earns the optimal revenue (pcmdp.second_stage). "
                  "One value: the ties do not change it. 0: below 1e-14. Blank: no result.\n")

        def ss_cell(s):
            if s is None:
                return ""
            lo, hi = s["min"]["g"], s["max"]["g"]
            f = (lambda x: "0" if x < 1e-14 else f"{x:.2g}")
            return f(hi) if hi <= lo * (1 + 1e-6) + 1e-14 else f"{f(lo)} - {f(hi)}"

        for g in gammas:
            cols = [p for p in GAMMA_PROTOCOLS if any((p, a, g) in ss for a in alphas)]
            md.append(f"### gamma {g:g}\n")
            md.append(table(["alpha"] + [LABEL[p] for p in cols],
                            [[f"{a:.2f}"] + [ss_cell(ss.get((p, a, g))) for p in cols]
                             for a in alphas]) + "\n")
        md.append("### Protocols with their own tie rule\n")
        md.append(table(["alpha"] + [LABEL[p] for p in TIE_PROTOCOLS],
                        [[f"{a:.2f}"] + [ss_cell(ss.get((p, a, 0.5 if p == "utb" else None)))
                                         for p in TIE_PROTOCOLS] for a in alphas]) + "\n")

    # Depth of the deep reorgs
    md.append("## Mean depth of the 11+ reorgs (range over alpha and gamma)\n")
    md.append("Majority reorgs of 11 blocks or more, under the policy that maximises "
              "their count (dc) or their total depth (dd); the cap bounds the depth.\n")
    rows = []
    for p in GAMMA_PROTOCOLS + TIE_PROTOCOLS:
        cells = []
        for o in ("dc", "dd"):
            ds = [deep_mean(r) for r in recs if r["protocol"] == p and r["objective"] == o]
            ds = [d for d in ds if d is not None]
            cells.append(f"{min(ds):.1f}-{max(ds):.1f}" if ds else "")
        rows.append([LABEL[p]] + cells)
    md.append(table(["protocol", "count-optimal (dc)", "depth-weighted (dd)"], rows) + "\n")
    return "\n".join(md)


def deep_mean(r, min_depth=11):
    """Mean depth of the majority reorgs of min_depth+ blocks, or None if there are none.

    The jsonl's depth rates stop at depth 40 but the profile's mean depth covers
    every depth, so take the total depth from the mean and remove the shallow part.
    """
    deep = r.get("profile.majority_deep_reorgs_per_block") or 0
    if deep < 1e-12:
        return None
    total = r["profile.majority_mean_depth"] * r["profile.majority_reorgs_per_block"]
    rates = r.get("profile.majority_depth_rates") or {}
    shallow = sum(int(d) * x for d, x in rates.items() if int(d) < min_depth)
    return (total - shallow) / deep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--second-stage", help="directory of second_stage.py outputs")
    ap.add_argument("--ss-jsonl", help="write the second-stage records here")
    ap.add_argument("--sop-dir", help="directory of MRL #146 SoP runs (sop_tight --honest-shares)")
    args = ap.parse_args()
    recs = load(args.out_dir)
    if args.sop_dir:
        recs += load_sop(args.sop_dir)
    with open(args.jsonl, "w") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    ss = {}
    if args.second_stage:
        ss, flat = load_second_stage(args.second_stage)
        if args.ss_jsonl:
            with open(args.ss_jsonl, "w") as f:
                for r in flat:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
    Path(args.md).write_text(build(recs, ss))
    print(f"{len(recs)} records, {len(ss)} second-stage")


if __name__ == "__main__":
    main()
