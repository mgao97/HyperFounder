#!/usr/bin/env bash
# Background launcher for the cross-domain hypergraph validation experiment.
set -u
cd /home/user/GSK/mgao/HyperFounder || exit 1
LOG=experiments/cross_domain_validation/results/run_cdv.log
mkdir -p experiments/cross_domain_validation/results
# single-threaded CPU run; fully detached so it survives the launching shell
nohup python -u experiments/cross_domain_validation/run_cdv.py > "$LOG" 2>&1 &
echo "launched pid $!  -> tail -f $LOG"
