#!/bin/bash
# =============================================================================
# Run multiple experiments with FULL PARALLELISM.
#
# All experiments for each house run simultaneously (no inter-experiment
# dependency). Since experiments use the same INPUT but different OUTPUT
# directories, they are completely independent.
#
#   house_1: exp016 ┐
#            exp017 ├── all in parallel
#            exp018 ├──
#            exp019 ┘
#   house_2: (same)
#   ...all running in parallel across the cluster
#
# This maximizes cluster utilization — with 4 experiments × 171 houses,
# up to 684 jobs can run simultaneously (limited only by available CPUs).
#
# Fast houses:  1 SLURM job per experiment (pre-analysis + M1 + M2 + reports + cleanup)
# Slow houses:  2 SLURM jobs per experiment (month array + post with M2/reports/cleanup)
#
# Aggregates run per-experiment after ALL houses finish that experiment.
# Cross-experiment comparison runs once at the very end.
#
# Usage:
#     bash scripts/run_all_experiments.sh                      # run all 4
#     bash scripts/run_all_experiments.sh exp017_phase_balance exp018_mad_clean  # specific ones
#
# =============================================================================

# ---- Experiments to run (in order) ----
if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=(
        "exp016_ma_detrend"
        "exp017_phase_balance"
        "exp018_mad_clean"
        "exp019_combined_norm"
    )
fi

# ---- Configuration ----
PROJECT_ROOT="/home/hilakese/ElectricPatterns_new"
DATA_DIR="${PROJECT_ROOT}/INPUT/HouseholdData"
LOG_DIR="${PROJECT_ROOT}/experiment_pipeline/logs"
MONTHS_PARALLEL=8

COMPLETED_HOUSES_FILE="${PROJECT_ROOT}/experiment_pipeline/completed_houses.txt"
ALL_EXPERIMENTS_DIR="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments"

# ---- Normalization map ----
declare -A NORM_MAP
NORM_MAP["exp016_ma_detrend"]="ma_detrend"
NORM_MAP["exp017_phase_balance"]="phase_balance"
NORM_MAP["exp018_mad_clean"]="mad_clean"
NORM_MAP["exp019_combined_norm"]="combined"

# ---- Shared timestamp ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "FULL PARALLEL multi-experiment runner"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Timestamp:   ${TIMESTAMP}"
echo "Date:        $(date)"
echo "============================================================"
echo ""

# =================================================================
# Phase 0: Create experiment directories + metadata for ALL experiments
# =================================================================
declare -A EXP_OUTPUTS
declare -A EXP_REPORTS

for exp in "${EXPERIMENTS[@]}"; do
    exp_output="${ALL_EXPERIMENTS_DIR}/${exp}_${TIMESTAMP}"
    reports_dir="${exp_output}/reports"
    mkdir -p "$exp_output" "$reports_dir"

    EXP_OUTPUTS[$exp]="$exp_output"
    EXP_REPORTS[$exp]="$reports_dir"

    # Initialize timing CSV
    echo "house_id,n_months,start_time,end_time,elapsed_seconds,elapsed_human,status" > "${exp_output}/house_timing.csv"

    # Save experiment metadata (once, from login node)
    cd "${PROJECT_ROOT}/experiment_pipeline"
    python -c "
import sys; sys.path.insert(0, 'src')
from core.config import get_experiment, save_experiment_metadata
exp = get_experiment('${exp}')
save_experiment_metadata(exp, '${exp_output}')
print('  Metadata: ${exp_output}')
"
    echo "  Created: ${exp_output}"
done
cd "${PROJECT_ROOT}"

# =================================================================
# Load completed houses
# =================================================================
declare -A COMPLETED
if [ -f "$COMPLETED_HOUSES_FILE" ]; then
    while IFS= read -r hid; do
        [[ -z "$hid" || "$hid" =~ ^[[:space:]]*# ]] && continue
        hid=$(echo "$hid" | xargs)
        COMPLETED["$hid"]=1
    done < "$COMPLETED_HOUSES_FILE"
    echo ""
    echo "Loaded ${#COMPLETED[@]} fast houses from completed_houses.txt"
else
    echo "ERROR: completed_houses.txt not found at $COMPLETED_HOUSES_FILE"
    exit 1
fi

# Helper: count unique months for a house
count_unique_months() {
    local house_dir="$1" house_id="$2"
    local -A seen_months
    local count=0
    for f in "$house_dir"/${house_id}_[0-9][0-9]_[0-9][0-9][0-9][0-9].pkl \
             "$house_dir"/${house_id}_[0-9][0-9]_[0-9][0-9][0-9][0-9].csv; do
        [ -f "$f" ] || continue
        local month_str
        month_str=$(basename "$f" | sed "s/^${house_id}_//" | sed 's/\..*//')
        if [[ -z "${seen_months[$month_str]}" ]]; then
            seen_months["$month_str"]=1
            count=$((count + 1))
        fi
    done
    echo "$count"
}

# Temp files to track job IDs per experiment (for aggregate dependencies)
for exp in "${EXPERIMENTS[@]}"; do
    : > "/tmp/final_jobs_${exp}_${TIMESTAMP}.txt"
done

echo ""
echo "============================================================"
echo "Submitting per-house chains..."
echo "============================================================"

HOUSE_COUNT=0
FAST_COUNT=0
SLOW_COUNT=0

# =================================================================
# Phase 1: Per-house parallel — each house runs all experiments simultaneously
# =================================================================
for house_dir in "$DATA_DIR"/*/; do
    house_id=$(basename "$house_dir")
    [[ "$house_id" =~ ^[0-9]+$ ]] || continue

    N_MONTHS=$(count_unique_months "$house_dir" "$house_id")
    [ "$N_MONTHS" -eq 0 ] && continue

    HOUSE_COUNT=$((HOUSE_COUNT + 1))

    is_fast=""
    [[ -n "${COMPLETED[$house_id]}" ]] && is_fast="1"

    for exp_idx in "${!EXPERIMENTS[@]}"; do
        exp="${EXPERIMENTS[$exp_idx]}"
        exp_num="${exp:3:3}"   # "016", "017", etc.
        exp_output="${EXP_OUTPUTS[$exp]}"
        reports_dir="${EXP_REPORTS[$exp]}"
        norm_method="${NORM_MAP[$exp]:-none}"

        if [ -n "$is_fast" ]; then
            # ==========================================================
            # FAST HOUSE: single job per experiment
            # Pre-analysis + M1 + M2 + reports + cleanup
            # ==========================================================
            PIPE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_f${exp_num}_${house_id}_XXXXXX.sh")

            cat > "$PIPE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=f${exp_num}_${house_id}
#SBATCH --output=${LOG_DIR}/f${exp_num}_${house_id}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/f${exp_num}_${house_id}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0

echo "========================================"
echo "House ${house_id} / ${exp} — FAST"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

# --- Pre-analysis ---
cd "${PROJECT_ROOT}/house_analysis"
python scripts/run_analysis.py \
    --houses ${house_id} \
    --input-dir ${DATA_DIR} \
    --output-dir ${reports_dir} \
    --publish house \
    --normalize ${norm_method}

# --- Pipeline (M1 + M2) ---
cd "${PROJECT_ROOT}/experiment_pipeline"

START_EPOCH=\$(date +%s)
START_TIME=\$(date '+%Y-%m-%d %H:%M:%S')

python -u scripts/test_single_house.py \
    --house_id ${house_id} \
    --experiment_name ${exp} \
    --output_path ${exp_output} \
    --skip_visualization \
    --minimal_output

EXIT_CODE=\$?

END_EPOCH=\$(date +%s)
END_TIME=\$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=\$((END_EPOCH - START_EPOCH))
HOURS=\$((ELAPSED / 3600))
MINS=\$(( (ELAPSED % 3600) / 60 ))
SECS=\$((ELAPSED % 60))
ELAPSED_HUMAN="\${HOURS}h \${MINS}m \${SECS}s"

if [ \$EXIT_CODE -eq 0 ]; then
    STATUS="OK"
else
    STATUS="FAIL(exit=\${EXIT_CODE})"
fi

echo "${house_id},${N_MONTHS},\$START_TIME,\$END_TIME,\$ELAPSED,\$ELAPSED_HUMAN,\$STATUS" >> ${exp_output}/house_timing.csv

echo "House ${house_id}: \$STATUS (\$ELAPSED_HUMAN)"

# --- Reports + cleanup (only on success) ---
if [ \$EXIT_CODE -eq 0 ]; then
    echo "Generating reports..."

    cd "${PROJECT_ROOT}/disaggregation_analysis"
    python scripts/run_dynamic_report.py \
        --experiment ${exp_output} \
        --houses ${house_id} \
        --output-dir ${reports_dir} \
        --publish segregation \
        2>&1
    echo "  Segregation report: exit \$?"

    cd "${PROJECT_ROOT}/identification_analysis"
    python scripts/run_identification_report.py \
        --experiment ${exp_output} \
        --houses ${house_id} \
        --output-dir ${reports_dir} \
        --publish identification \
        2>&1
    echo "  Identification report: exit \$?"

    echo "Cleaning up pkl files..."
    cd "${PROJECT_ROOT}/experiment_pipeline"
    python -c "
import sys; sys.path.insert(0, 'src')
from identification.cleanup import cleanup_after_reports
from pathlib import Path
r = cleanup_after_reports(Path('${exp_output}'), '${house_id}')
print(f'  Cleanup: {r[\"dirs_deleted\"]} directories removed')
"
fi

echo "End: \$(date)"
EOF

            PIPE_JOB=$(sbatch "$PIPE_SCRIPT" | awk '{print $4}')
            rm -f "$PIPE_SCRIPT"

            echo "${PIPE_JOB}" >> "/tmp/final_jobs_${exp}_${TIMESTAMP}.txt"

        else
            # ==========================================================
            # SLOW HOUSE: month-level parallel + post job
            # Array: M1 per month
            # Post:  pre-analysis + M2 + reports + cleanup
            # ==========================================================

            # --- Array job (M1 per month) ---
            ARRAY_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_m${exp_num}_${house_id}_XXXXXX.sh")

            cat > "$ARRAY_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=m${exp_num}_${house_id}
#SBATCH --output=${LOG_DIR}/m${exp_num}_${house_id}_${TIMESTAMP}_%A_%a.out
#SBATCH --error=${LOG_DIR}/m${exp_num}_${house_id}_${TIMESTAMP}_%A_%a.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --array=0-$((N_MONTHS - 1))%${MONTHS_PARALLEL}

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"
python -u scripts/run_single_month.py \
    --house_id ${house_id} \
    --month_index \$SLURM_ARRAY_TASK_ID \
    --experiment_name ${exp} \
    --output_path ${exp_output} \
    --input_path ${DATA_DIR}
EOF

            ARRAY_JOB=$(sbatch "$ARRAY_SCRIPT" | awk '{print $4}')
            rm -f "$ARRAY_SCRIPT"

            # --- Post job (pre-analysis + M2 + reports + cleanup) ---
            POST_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_p${exp_num}_${house_id}_XXXXXX.sh")

            cat > "$POST_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=p${exp_num}_${house_id}
#SBATCH --output=${LOG_DIR}/p${exp_num}_${house_id}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/p${exp_num}_${house_id}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterok:${ARRAY_JOB}

echo "========================================"
echo "House ${house_id} / ${exp} — SLOW (post)"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Depends on array: ${ARRAY_JOB}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

# --- Pre-analysis ---
cd "${PROJECT_ROOT}/house_analysis"
python scripts/run_analysis.py \
    --houses ${house_id} \
    --input-dir ${DATA_DIR} \
    --output-dir ${reports_dir} \
    --publish house \
    --normalize ${norm_method}

# --- M2 Identification ---
cd "${PROJECT_ROOT}/experiment_pipeline"

START_EPOCH=\$(date +%s)
START_TIME=\$(date '+%Y-%m-%d %H:%M:%S')

python -u scripts/run_identification.py \
    --experiment_dir ${exp_output} \
    --house_id ${house_id}

EXIT_CODE=\$?

END_EPOCH=\$(date +%s)
END_TIME=\$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=\$((END_EPOCH - START_EPOCH))
HOURS=\$((ELAPSED / 3600))
MINS=\$(( (ELAPSED % 3600) / 60 ))
SECS=\$((ELAPSED % 60))
ELAPSED_HUMAN="\${HOURS}h \${MINS}m \${SECS}s"

if [ \$EXIT_CODE -eq 0 ]; then
    STATUS="OK-MONTHLY"
else
    STATUS="FAIL-MONTHLY(exit=\${EXIT_CODE})"
fi

echo "${house_id},${N_MONTHS},\$START_TIME,\$END_TIME,\$ELAPSED,\$ELAPSED_HUMAN,\$STATUS" >> ${exp_output}/house_timing.csv

echo "House ${house_id}: \$STATUS (\$ELAPSED_HUMAN)"

# --- Reports + cleanup (only on success) ---
if [ \$EXIT_CODE -eq 0 ]; then
    echo "Generating reports..."

    cd "${PROJECT_ROOT}/disaggregation_analysis"
    python scripts/run_dynamic_report.py \
        --experiment ${exp_output} \
        --houses ${house_id} \
        --output-dir ${reports_dir} \
        --publish segregation \
        2>&1
    echo "  Segregation report: exit \$?"

    cd "${PROJECT_ROOT}/identification_analysis"
    python scripts/run_identification_report.py \
        --experiment ${exp_output} \
        --houses ${house_id} \
        --output-dir ${reports_dir} \
        --publish identification \
        2>&1
    echo "  Identification report: exit \$?"

    echo "Cleaning up pkl files..."
    cd "${PROJECT_ROOT}/experiment_pipeline"
    python -c "
import sys; sys.path.insert(0, 'src')
from identification.cleanup import cleanup_after_reports
from pathlib import Path
r = cleanup_after_reports(Path('${exp_output}'), '${house_id}')
print(f'  Cleanup: {r[\"dirs_deleted\"]} directories removed')
"
fi

echo "End: \$(date)"
EOF

            POST_JOB=$(sbatch "$POST_SCRIPT" | awk '{print $4}')
            rm -f "$POST_SCRIPT"

            echo "${POST_JOB}" >> "/tmp/final_jobs_${exp}_${TIMESTAMP}.txt"
        fi
    done

    # Print house summary
    if [ -n "$is_fast" ]; then
        echo "  House ${house_id} (${N_MONTHS}m) [FAST] — ${#EXPERIMENTS[@]} experiments in parallel"
        FAST_COUNT=$((FAST_COUNT + 1))
    else
        echo "  House ${house_id} (${N_MONTHS}m) [SLOW] — ${#EXPERIMENTS[@]} experiments in parallel (monthly parallel)"
        SLOW_COUNT=$((SLOW_COUNT + 1))
    fi
done

echo ""
echo "============================================================"
echo "Submitted ${HOUSE_COUNT} houses (fast=${FAST_COUNT}, slow=${SLOW_COUNT})"
echo "Each house runs ${#EXPERIMENTS[@]} experiments IN PARALLEL"
echo ""

# =================================================================
# Phase 2: Aggregate reports — per experiment
#
# For each experiment, submit aggregate jobs that depend on ALL
# houses finishing that experiment.
# =================================================================
echo "Submitting aggregate reports..."
echo "============================================================"

ALL_IDENT_JOBS=()

for exp in "${EXPERIMENTS[@]}"; do
    exp_num="${exp:3:3}"
    exp_output="${EXP_OUTPUTS[$exp]}"
    reports_dir="${EXP_REPORTS[$exp]}"

    FINAL_DEPS=$(cat "/tmp/final_jobs_${exp}_${TIMESTAMP}.txt" | tr '\n' ':' | sed 's/:$//')

    if [ -z "$FINAL_DEPS" ]; then
        echo "  WARNING: No jobs for ${exp}, skipping aggregates"
        continue
    fi

    # --- Aggregate house pre-analysis ---
    AGG_HOUSE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_ah_${exp_num}_XXXXXX.sh")
    cat > "$AGG_HOUSE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ah_${exp_num}
#SBATCH --output=${LOG_DIR}/ah_${exp_num}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/ah_${exp_num}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${FINAL_DEPS}

echo "AGGREGATE: House pre-analysis — ${exp}"
echo "Start: \$(date)"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/house_analysis"
python scripts/run_analysis.py \
    --output-dir ${reports_dir} \
    --publish house \
    --aggregate-only \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
    AGG_HOUSE_JOB=$(sbatch "$AGG_HOUSE_SCRIPT" | awk '{print $4}')
    rm -f "$AGG_HOUSE_SCRIPT"

    # --- Aggregate segregation ---
    AGG_SEG_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_as_${exp_num}_XXXXXX.sh")
    cat > "$AGG_SEG_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=as_${exp_num}
#SBATCH --output=${LOG_DIR}/as_${exp_num}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/as_${exp_num}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${FINAL_DEPS}:${AGG_HOUSE_JOB}

echo "AGGREGATE: Segregation — ${exp}"
echo "Start: \$(date)"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/disaggregation_analysis"
python scripts/run_dynamic_report.py \
    --experiment ${exp_output} \
    --output-dir ${reports_dir} \
    --publish segregation \
    --aggregate-only \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
    AGG_SEG_JOB=$(sbatch "$AGG_SEG_SCRIPT" | awk '{print $4}')
    rm -f "$AGG_SEG_SCRIPT"

    # --- Identification ALL houses ---
    IDENT_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_id_${exp_num}_XXXXXX.sh")
    cat > "$IDENT_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=id_${exp_num}
#SBATCH --output=${LOG_DIR}/id_${exp_num}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/id_${exp_num}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${AGG_SEG_JOB}

echo "IDENTIFICATION: All houses — ${exp}"
echo "Start: \$(date)"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/identification_analysis"
python scripts/run_identification_report.py \
    --experiment ${exp_output} \
    --output-dir ${reports_dir} \
    --publish identification \
    --aggregate-only \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
    IDENT_JOB=$(sbatch "$IDENT_SCRIPT" | awk '{print $4}')
    rm -f "$IDENT_SCRIPT"

    ALL_IDENT_JOBS+=($IDENT_JOB)
    echo "  ${exp}: agg_house=${AGG_HOUSE_JOB}, agg_seg=${AGG_SEG_JOB}, ident=${IDENT_JOB}"

    # Cleanup temp file
    rm -f "/tmp/final_jobs_${exp}_${TIMESTAMP}.txt"
done

# =================================================================
# Phase 3: Cross-experiment comparison (after ALL experiments done)
# =================================================================
if [ ${#ALL_IDENT_JOBS[@]} -gt 0 ]; then
    IDENT_DEPS=$(IFS=:; echo "${ALL_IDENT_JOBS[*]}")
    LAST_EXP="${EXPERIMENTS[-1]}"
    LAST_REPORTS="${EXP_REPORTS[$LAST_EXP]}"

    COMPARE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_compare_XXXXXX.sh")
    cat > "$COMPARE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=compare_all
#SBATCH --output=${LOG_DIR}/compare_all_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/compare_all_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${IDENT_DEPS}

echo "CROSS-EXPERIMENT COMPARISON"
echo "Start: \$(date)"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"
python scripts/compare_experiments.py \
    --scan \
    --scan-dir ${ALL_EXPERIMENTS_DIR} \
    --output-dir ${LAST_REPORTS}/comparison \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
    COMPARE_JOB=$(sbatch "$COMPARE_SCRIPT" | awk '{print $4}')
    rm -f "$COMPARE_SCRIPT"
    echo ""
    echo "  Cross-experiment comparison: ${COMPARE_JOB} (after all experiments)"
fi

echo ""
echo "============================================================"
echo "All submitted!"
echo ""
echo "  Houses:      ${HOUSE_COUNT} (fast=${FAST_COUNT}, slow=${SLOW_COUNT})"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "  Strategy:    FULL PARALLEL (all experiments simultaneously per house)"
echo ""
echo "  All ${#EXPERIMENTS[@]} experiments run simultaneously for each house."
echo "  Total jobs: up to ${HOUSE_COUNT} x ${#EXPERIMENTS[@]} running at once."
echo ""
for exp in "${EXPERIMENTS[@]}"; do
    echo "  ${exp}:"
    echo "    Output:  ${EXP_OUTPUTS[$exp]}"
    echo "    Reports: ${EXP_REPORTS[$exp]}"
done
echo ""
echo "Monitor:"
echo "  squeue -u \$USER                          # all jobs"
echo "  squeue -u \$USER | grep f016              # fast houses exp016"
echo "  squeue -u \$USER | grep m017              # slow houses exp017 months"
echo "  squeue -u \$USER | grep -E 'ah_|as_|id_'  # aggregate jobs"
echo "============================================================"
