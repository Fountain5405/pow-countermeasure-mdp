#!/bin/bash
cd ~/mdp_work/blind/work
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$HOME/mdp_work/blind/code
PY=~/mdp_work/venv/bin/python; ulimit -v 64000000
echo "=== cross2 $(date)"; nice $PY -s py/q2_cross2.py > q2_cross2.log 2>&1
echo "=== shtb r15 mF10 grid $(date)"; nice $PY -s py/q2_exact.py exp/shtb_r15_mF10 4 0 grid 0.0575 0.058 0.0585 0.06 0.065 0.10 > q2_shtb_r15_mF10_grid.log 2>&1
echo "=== shtb r15 mF20 grid2 $(date)"; nice $PY -s py/q2_exact.py exp/shtb_r15_mF20 4 0 grid 0.0575 0.058 0.0585 0.06 > q2_shtb_r15_mF20_grid2.log 2>&1
echo "=== chainA2 done $(date)"
