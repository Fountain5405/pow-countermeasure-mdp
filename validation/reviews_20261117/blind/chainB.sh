#!/bin/bash
cd ~/mdp_work/blind/work
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$HOME/mdp_work/blind/code
PY=~/mdp_work/venv/bin/python; ulimit -v 64000000
echo "=== q3 tiny $(date)"; (cd py && nice $PY -s q3_tiny.py > ../q3_tiny.log 2>&1)
echo "=== pop mB20 grid $(date)"; nice $PY -s py/q2_exact.py exp/pop_mB20 5 37127 grid 0.26400 0.26405 0.30 0.35 0.40 > q2_pop_mB20_grid.log 2>&1
echo "=== chainB done $(date)"
