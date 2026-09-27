#!/usr/bin/env bash

set -u

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sweep_script="experiments/pose2pose_unicycle/param_sweep_pose2pose_unicycle_simple_initial_guess.py"
overall_status=0

cd "${repository_root}"

# MA27 is provided by the locally built Coin-HSL library. User services do not
# inherit the interactive shell's LD_LIBRARY_PATH, so expose it explicitly.
export LD_LIBRARY_PATH="/home/sonia/coinhsl-2024.05.15:/home/sonia/coinhsl-2024.05.15/build:${LD_LIBRARY_PATH:-}"

echo "Sequence started: $(date --iso-8601=seconds)"
echo "Configuration: sweeps=2,3, N=100, M=4, linear solver=ma27, initial guess=TST"

for sweep_id in 2 3; do
    echo
    echo "======================================================================"
    echo "Starting sweep ${sweep_id}: $(date --iso-8601=seconds)"
    echo "======================================================================"

    if MPLCONFIGDIR=/tmp/kappa-mpl \
       PYTHONPATH=src \
       PYTHONUNBUFFERED=1 \
       .venv/bin/python "${sweep_script}" \
           --sweep-id "${sweep_id}" \
           --N 100 \
           --M 4 \
           --linear-solver ma27; then
        echo "Sweep ${sweep_id} completed: $(date --iso-8601=seconds)"
    else
        status=$?
        overall_status=1
        echo "Sweep ${sweep_id} failed with exit code ${status}: $(date --iso-8601=seconds)"
    fi
done

echo
echo "Sequence finished: $(date --iso-8601=seconds)"
exit "${overall_status}"
