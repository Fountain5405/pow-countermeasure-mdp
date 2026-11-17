#!/bin/bash
cd ~/mdp_work/blind/work
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$HOME/mdp_work/blind/code
PY=~/mdp_work/venv/bin/python
ulimit -v 64000000
echo "=== pop mB16 bisect $(date)"; nice $PY -s py/q2_exact.py exp/pop_mB16 5 19719 bisect 0.2635 0.2650 > q2_pop_mB16_bisect.log 2>&1
while [ ! -s exp/shtb_r5_mF40_a0.40_Rh4.bin ]; do sleep 20; done; sleep 10
echo "=== shtb r15 bisect $(date)"; nice $PY -s py/q2_exact.py exp/shtb_r15_mF40 4 0 bisect 0.02 0.10 > q2_shtb_r15_bisect.log 2>&1
echo "=== shtb r10 bisect $(date)"; nice $PY -s py/q2_exact.py exp/shtb_r10_mF40 4 0 bisect 0.02 0.15 > q2_shtb_r10_bisect.log 2>&1
echo "=== shtb r5 bisect $(date)"; nice $PY -s py/q2_exact.py exp/shtb_r5_mF40 4 0 bisect 0.05 0.30 > q2_shtb_r5_bisect.log 2>&1
echo "=== q3 tiny $(date)"; cd py && nice $PY -s q3_tiny.py > ../q3_tiny.log 2>&1; cd ..
while [ ! -s exp/pop_mB20_a0.40_Rh5.bin ]; do sleep 20; done; sleep 10
echo "=== pop mB20 grid $(date)"; nice $PY -s py/q2_exact.py exp/pop_mB20 5 37127 grid 0.30 0.35 0.40 > q2_pop_mB20_grid.log 2>&1
echo "=== pop mB20 bisect $(date)"; nice $PY -s py/q2_exact.py exp/pop_mB20 5 37127 bisect 0.2635 0.2650 > q2_pop_mB20_bisect.log 2>&1
echo "=== chain1 done $(date)"
