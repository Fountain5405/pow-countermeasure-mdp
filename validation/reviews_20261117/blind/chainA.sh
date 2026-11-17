#!/bin/bash
cd ~/mdp_work/blind/work
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$HOME/mdp_work/blind/code
PY=~/mdp_work/venv/bin/python; ulimit -v 64000000
echo "=== export shtb mF10/20 $(date)"; (cd zp/shtb && nice $HOME/mdp_work/bin/micromamba run -p $HOME/mdp_work/octave_env octave-cli --no-gui export_shtb2.m > ~/mdp_work/blind/work/export_shtb2.log 2>&1)
echo "=== shtb r15 mF10 bisect $(date)"; nice $PY -s py/q2_exact.py exp/shtb_r15_mF10 4 0 bisect 0.02 0.10 > q2_shtb_r15_mF10_bisect.log 2>&1
echo "=== shtb r10 mF10 bisect $(date)"; nice $PY -s py/q2_exact.py exp/shtb_r10_mF10 4 0 bisect 0.02 0.15 > q2_shtb_r10_mF10_bisect.log 2>&1
echo "=== shtb r5 mF10 bisect $(date)"; nice $PY -s py/q2_exact.py exp/shtb_r5_mF10 4 0 bisect 0.05 0.30 > q2_shtb_r5_mF10_bisect.log 2>&1
echo "=== shtb r15 mF20 grid $(date)"; nice $PY -s py/q2_exact.py exp/shtb_r15_mF20 4 0 grid 0.064 0.0645 0.065 0.0655 0.10 > q2_shtb_r15_mF20_grid.log 2>&1
echo "=== cross $(date)"; nice $PY -s py/q2_cross.py > q2_cross.log 2>&1
echo "=== chainA done $(date)"
