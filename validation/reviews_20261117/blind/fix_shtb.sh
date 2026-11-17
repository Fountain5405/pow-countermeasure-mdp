#!/bin/bash
cd ~/mdp_work/blind/work
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$HOME/mdp_work/blind/code
ulimit -v 64000000
nice ~/mdp_work/venv/bin/python -s py/q1_mf30.py > q1_mf30.log 2>&1
cd zp/shtb && nice $HOME/mdp_work/bin/micromamba run -p $HOME/mdp_work/octave_env octave-cli --no-gui export_shtb.m > ~/mdp_work/blind/work/export_shtb.log 2>&1
