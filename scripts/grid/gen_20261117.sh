#!/bin/bash
# Job lists for the 2026-11-17 grid (Rucknium's gist): every protocol, three
# objectives (revenue; disruption = reorgs of 11+ blocks per block interval,
# counted and depth-weighted), alpha 0.10-0.48, gamma 0 / 0.05 / 0.5 / 1 where
# the protocol has one. Caps are the references' (solve.py defaults) except
# Nakamoto at Zhang & Preneel's lmax 160; the SoP tight model uses the caps of
# the published tight results (x +-40, pending 24), disruption count only.
# Writes jobs_light.txt and jobs_heavy.txt (heavy: the 1-2 h, 15-25 GB solves,
# longest first). One line per job: <name> <module> <args...>.
A="0.10 0.15 0.20 0.25 0.30 0.33 0.35 0.38 0.40 0.42 0.45 0.48"
G4="0 0.05 0.5 1"
OBJ_rev="--objective revenue"
OBJ_dc="--objective disruption --normalization time --weight count"
OBJ_dd="--objective disruption --normalization time --weight depth"

emit() {  # <name prefix> <solve.py protocol args...>: one job per objective
    local name=$1; shift
    for o in rev dc dd; do
        local v="OBJ_$o"
        echo "${name}_$o pcmdp.solve $* ${!v} --solver howard"
    done
}

light() {
    for a in $A; do
        for g in $G4; do
            emit "nakamoto_a${a}_g$g" nakamoto --alpha "$a" --gamma "$g" --max-fork 160
            emit "rs_a${a}_g$g" rs --alpha "$a" --gamma "$g"
            emit "prs_a${a}_g$g" prs --alpha "$a" --gamma "$g"
            emit "subchains_a${a}_g$g" subchains --alpha "$a" --gamma "$g"
        done
        emit "utb_a${a}_g0.5" utb --alpha "$a" --gamma 0.5
        emit "udtb_a$a" udtb --alpha "$a"
    done
}

heavy() {
    for a in $A; do
        for g in 0 1; do
            emit "fruitchains_a${a}_g$g" fruitchains --alpha "$a" --gamma "$g"
        done
    done
    for a in $A; do
        echo "soptight_a${a}_dc pcmdp.sop_tight --alpha $a --x-min -40 --x-max 40 --p-max 24 --q-max 24"
    done
    for a in $A; do
        emit "shtb_a$a" shtb --alpha "$a"
        emit "pop_a$a" pop --alpha "$a"
    done
}

light > jobs_light.txt
heavy > jobs_heavy.txt
wc -l jobs_light.txt jobs_heavy.txt
