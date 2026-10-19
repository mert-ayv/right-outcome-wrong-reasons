#!/bin/bash
# Wrapper for `launchctl submit` (launchd restarts it whenever it exits). Stops itself once the run is done.
cd /Users/mertayvazoglu/codex/pilot || exit 1
if [ -f results/rerun_mlx/DONE ]; then launchctl remove com.mert.h1rerun; exit 0; fi
echo "=== start $(date)" >> logs/rerun_attempts.log
/usr/bin/caffeinate -i ./run_rerun_mlx.sh >> logs/rerun_mlx.log 2>&1
code=$?
echo "=== exit $code $(date)" >> logs/rerun_attempts.log
if [ $code -eq 0 ]; then touch results/rerun_mlx/DONE; fi
sleep 20
