#!/bin/bash
# =============================================================================
# Run multiple experiments SEQUENTIALLY on SLURM.
#
# Each experiment waits for the previous one to fully complete (including
# reports and cleanup) before starting. This prevents disk quota issues
# from running experiments in parallel.
#
# The aggressive cleanup (cleanup_after_reports) runs after each house's
# reports are generated, keeping only JSONs/CSVs/HTMLs and deleting
# heavy pkl files. This reduces each experiment from ~35G to ~4G.
#
# Usage:
#     bash scripts/run_all_experiments.sh
#
# To customize which experiments to run, edit the EXPERIMENTS array below.
# =============================================================================

# ---- Experiments to run (in order) ----
EXPERIMENTS=(
    "exp016_ma_detrend"
    "exp017_phase_balance"
    "exp018_mad_clean"
    "exp019_combined_norm"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAST_JOB=""

echo "============================================================"
echo "Sequential experiment runner"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Date: $(date)"
echo "============================================================"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    n=$((i + 1))
    total=${#EXPERIMENTS[@]}

    echo "[$n/$total] Submitting: $exp"

    # Write output to temp file AND show it in real-time via tee
    TMPOUT=$(mktemp /tmp/exp_submit_XXXXXX.log)

    if [ -n "$LAST_JOB" ]; then
        echo "  Chaining after job $LAST_JOB"
        bash "$SCRIPT_DIR/sbatch_run_houses.sh" "$exp" "$LAST_JOB" 2>&1 | tee "$TMPOUT"
    else
        bash "$SCRIPT_DIR/sbatch_run_houses.sh" "$exp" 2>&1 | tee "$TMPOUT"
    fi

    # Extract the last job ID for chaining
    LAST_JOB=$(grep "^LAST_JOB_ID=" "$TMPOUT" | tail -1 | cut -d= -f2)
    rm -f "$TMPOUT"

    if [ -z "$LAST_JOB" ]; then
        echo "ERROR: Could not extract LAST_JOB_ID from $exp output. Stopping."
        exit 1
    fi

    echo ""
    echo "  -> $exp final job: $LAST_JOB"
    echo ""
done

echo "============================================================"
echo "All ${#EXPERIMENTS[@]} experiments submitted!"
echo "Last job in chain: $LAST_JOB"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER                    # running/pending jobs"
echo "  squeue -u \$USER | wc -l            # count jobs"
echo "  sacct --starttime today -X --format=JobName,State,Elapsed | tail -20"
echo "============================================================"
