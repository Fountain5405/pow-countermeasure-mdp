#!/bin/bash
# Run one grid job: run1.sh <name> <module> <args...>, from the grid directory
# (which holds code/, out/, pol/ and ../venv). A job whose out/<name>.json
# exists is skipped, so a queue can be restarted. One status line per job goes
# to STATUS.
cd "$(dirname "$0")" || exit 1
name=$1; module=$2; shift 2
[ -s "out/$name.json" ] && exit 0
ulimit -v "${MEM_KB:-64000000}"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=code \
    timeout "${TMO:-86400}" nice ../venv/bin/python -m "$module" "$@" --save-policy "pol/$name.npz" \
    > "out/$name.json.tmp" 2> "out/$name.err"
rc=$?
[ $rc -eq 0 ] && mv "out/$name.json.tmp" "out/$name.json"
echo "$name exit $rc $(date -u +%FT%TZ)" >> STATUS
