#!/bin/bash
# =============================================================================
# Run multiple experiments in parallel using a manifest-based SLURM array.
#
# Instead of running one experiment at a time (sbatch_run_houses.sh), this
# script runs ALL specified experiments simultaneously. Each (experiment, house)
# pair is a single SLURM array task that runs:
#   1. Pre-analysis (with experiment-specific normalization)
#   2. Pipeline (M1 + M2)
#   3. Per-house reports (segregation + identification)
#
# Execution order:
#   Phase 1 — SLURM array (all experiments x all houses):
#     One big array job with --array=0-N%MAX_CONCURRENT.
#     Each task reads its (experiment, house, norm_method) from a manifest CSV
#     and runs pre-analysis + pipeline + per-house reports.
#
#   Phase 2 — Per-experiment aggregates (after array completes):
#     For EACH experiment:
#       agg_pre  → aggregate house pre-analysis report
#       agg_seg  → aggregate segregation report (depends on agg_pre)
#       ident_all → identification report for all houses (depends on agg_seg)
#
#   Phase 3 — Cross-experiment comparison (after all ident_all jobs):
#     Scans ALL experiments in OUTPUT/experiments/ and generates
#     comparison CSV + HTML.
#
# Dependency chain (per experiment):
#   array ──→ agg_pre ──→ agg_seg ──→ ident_all ──→ compare
#
# All experiments share the same timestamp (TS) set once at script start.
# Output structure per experiment:
#   {experiment}_{TS}/
#     ├── run_0/ ... run_3/        — Pipeline output per iteration
#     ├── device_sessions/         — M2 output (JSON per house)
#     ├── house_timing.csv         — Timing log
#     └── reports/
#         ├── house_reports/       — Per-house pre-analysis
#         ├── segregation_reports/ — Per-house segregation
#         └── identification_reports/ — Per-house identification
#
# Usage:
#     bash scripts/sbatch_run_all_experiments.sh
# =============================================================================

set -euo pipefail

# ---- Configuration ----
PROJECT_ROOT="/home/hilakese/ElectricPatterns_new"
DATA_DIR="${PROJECT_ROOT}/INPUT/HouseholdData"
LOG_DIR="${PROJECT_ROOT}/experiment_pipeline/logs"
ALL_EXPERIMENTS_DIR="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments"

# Max concurrent SLURM array tasks (prevents job queue overflow)
MAX_CONCURRENT=300

# ---- Experiments to run: "config_name:norm_method" ----
# Edit this list to add/remove experiments.
# norm_method must match a valid normalization method in core/normalization.py.
EXPERIMENTS=(
    "exp016_ma_detrend:ma_detrend"
    "exp017_phase_balance:phase_balance"
    "exp018_mad_clean:mad_clean"
    "exp019_combined_norm:combined"
)
# Uncomment to include baseline (no normalization) for comparison:
# EXPERIMENTS+=("exp015_hole_repair:none")

# ---- Shared timestamp (set ONCE — all experiments use the same TS) ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

# ---- Discover house IDs ----
HOUSE_IDS=()
for house_dir in "$DATA_DIR"/*/; do
    house_id=$(basename "$house_dir")
    # Skip non-numeric directories
    [[ "$house_id" =~ ^[0-9]+$ ]] || continue
    # Check that the house has data files
    has_files=false
    for f in "$house_dir"/${house_id}_[0-9][0-9]_[0-9][0-9][0-9][0-9].*; do
        [ -f "$f" ] && has_files=true && break
    done
    $has_files && HOUSE_IDS+=("$house_id")
done

N_HOUSES=${#HOUSE_IDS[@]}
N_EXPERIMENTS=${#EXPERIMENTS[@]}

if [ "$N_HOUSES" -eq 0 ]; then
    echo "ERROR: No houses found in $DATA_DIR"
    exit 1
fi

echo "============================================================"
echo "Multi-Experiment Parallel Runner"
echo "============================================================"
echo "Experiments:    $N_EXPERIMENTS"
echo "Houses:         $N_HOUSES"
echo "Total tasks:    $((N_EXPERIMENTS * N_HOUSES))"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Timestamp:      $TIMESTAMP"
echo ""
for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config_name norm_method <<< "$exp_entry"
    echo "  - ${config_name} (norm=${norm_method})"
done
echo ""

# ============================================================
# SETUP: Create output dirs + experiment metadata
# ============================================================
echo "Setting up experiment directories..."

# Associative array: config_name -> output_dir
declare -A EXP_OUTPUTS

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config_name norm_method <<< "$exp_entry"

    output_dir="${ALL_EXPERIMENTS_DIR}/${config_name}_${TIMESTAMP}"
    reports_dir="${output_dir}/reports"

    mkdir -p "$output_dir" "$reports_dir"

    # Initialize timing CSV header
    echo "house_id,n_months,start_time,end_time,elapsed_seconds,elapsed_human,status" \
        > "${output_dir}/house_timing.csv"

    # Pre-create experiment metadata (once, from login node — avoids race condition
    # when multiple SLURM tasks call save_experiment_metadata simultaneously)
    cd "${PROJECT_ROOT}/experiment_pipeline"
    python -c "
import sys; sys.path.insert(0, 'src')
from core.config import get_experiment, save_experiment_metadata
exp = get_experiment('${config_name}')
save_experiment_metadata(exp, '${output_dir}')
print('  ${config_name} -> ${output_dir}')
"
    cd "$PROJECT_ROOT"

    EXP_OUTPUTS["$config_name"]="$output_dir"
done
echo ""

# ============================================================
# GENERATE MANIFEST CSV
# ============================================================
# Each line: config_name,house_id,output_dir,norm_method
# Task index = line number (0-based)
MANIFEST="${LOG_DIR}/manifest_${TIMESTAMP}.csv"

TASK_COUNT=0
for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config_name norm_method <<< "$exp_entry"
    output_dir="${EXP_OUTPUTS[$config_name]}"

    for house_id in "${HOUSE_IDS[@]}"; do
        echo "${config_name},${house_id},${output_dir},${norm_method}" >> "$MANIFEST"
        TASK_COUNT=$((TASK_COUNT + 1))
    done
done

echo "Manifest: ${MANIFEST}"
echo "  ${TASK_COUNT} tasks (${N_EXPERIMENTS} experiments x ${N_HOUSES} houses)"
echo ""

# ============================================================
# PHASE 1: Submit SLURM array (all experiments x all houses)
# ============================================================
ARRAY_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_array_XXXXXX.sh")

cat > "$ARRAY_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ep_multi
#SBATCH --output=${LOG_DIR}/task_${TIMESTAMP}_%A_%a.out
#SBATCH --error=${LOG_DIR}/task_${TIMESTAMP}_%A_%a.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --array=0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}

echo "========================================"
echo "MANIFEST TASK \$SLURM_ARRAY_TASK_ID / $((TASK_COUNT - 1))"
echo "Array job: \$SLURM_ARRAY_JOB_ID[\$SLURM_ARRAY_TASK_ID]"
echo "Node: \$SLURM_JOB_NODELIST"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"

python -u scripts/run_manifest_task.py \\
    --manifest ${MANIFEST} \\
    --task-index \$SLURM_ARRAY_TASK_ID \\
    --input-dir ${DATA_DIR}

EXIT_CODE=\$?
echo "Task \$SLURM_ARRAY_TASK_ID exit: \$EXIT_CODE — End: \$(date)"
exit \$EXIT_CODE
EOF

ARRAY_JOB_ID=$(sbatch "$ARRAY_SCRIPT" | awk '{print $4}')
rm -f "$ARRAY_SCRIPT"

echo "Phase 1: Array job ${ARRAY_JOB_ID}"
echo "  ${TASK_COUNT} tasks, max ${MAX_CONCURRENT} concurrent"
echo ""

# ============================================================
# PHASE 2: Per-experiment aggregate reports
#
# For each experiment, submit a chain:
#   agg_pre → agg_seg → ident_all
#
# All depend on the ENTIRE array completing (afterany).
# Within each experiment, the chain is sequential.
# Across experiments, the chains run in parallel.
# ============================================================
echo "Phase 2: Submitting per-experiment aggregates..."

ALL_IDENT_JOBS=()

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config_name norm_method <<< "$exp_entry"
    exp_output="${EXP_OUTPUTS[$config_name]}"
    exp_reports="${exp_output}/reports"

    # --- agg_pre: Aggregate house pre-analysis ---
    AGG_PRE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_pre_${config_name}_XXXXXX.sh")
    cat > "$AGG_PRE_SCRIPT" << AGGPRE_EOF
#!/bin/bash
#SBATCH --job-name=agg_pre_${config_name}
#SBATCH --output=${LOG_DIR}/agg_pre_${config_name}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/agg_pre_${config_name}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${ARRAY_JOB_ID}

echo "========================================"
echo "AGGREGATE PRE-ANALYSIS: ${config_name}"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

EXPECTED=${N_HOUSES}
ACTUAL=\$(ls ${exp_reports}/house_reports/house_*.html 2>/dev/null | wc -l)
echo "Per-house reports: \${ACTUAL} / \${EXPECTED}"

if [ "\$ACTUAL" -lt "\$EXPECTED" ]; then
    echo "WARNING: Missing \$((EXPECTED - ACTUAL)) house reports."
fi

cd "${PROJECT_ROOT}/house_analysis"
python scripts/run_analysis.py \\
    --output-dir ${exp_reports} \\
    --publish house \\
    --aggregate-only \\
    2>&1

echo "Exit: \$? — End: \$(date)"
AGGPRE_EOF
    AGG_PRE_JOB=$(sbatch "$AGG_PRE_SCRIPT" | awk '{print $4}')
    rm -f "$AGG_PRE_SCRIPT"

    # --- agg_seg: Aggregate segregation ---
    AGG_SEG_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_seg_${config_name}_XXXXXX.sh")
    cat > "$AGG_SEG_SCRIPT" << AGGSEG_EOF
#!/bin/bash
#SBATCH --job-name=agg_seg_${config_name}
#SBATCH --output=${LOG_DIR}/agg_seg_${config_name}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/agg_seg_${config_name}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${ARRAY_JOB_ID}:${AGG_PRE_JOB}

echo "========================================"
echo "AGGREGATE SEGREGATION: ${config_name}"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

EXPECTED=${N_HOUSES}
ACTUAL=\$(ls ${exp_reports}/segregation_reports/house_*.html 2>/dev/null | wc -l)
echo "Per-house reports: \${ACTUAL} / \${EXPECTED}"

if [ "\$ACTUAL" -lt "\$EXPECTED" ]; then
    echo "WARNING: Missing \$((EXPECTED - ACTUAL)) segregation reports."
fi

cd "${PROJECT_ROOT}/disaggregation_analysis"
python scripts/run_dynamic_report.py \\
    --experiment ${exp_output} \\
    --output-dir ${exp_reports} \\
    --publish segregation \\
    --aggregate-only \\
    2>&1

echo "Exit: \$? — End: \$(date)"
AGGSEG_EOF
    AGG_SEG_JOB=$(sbatch "$AGG_SEG_SCRIPT" | awk '{print $4}')
    rm -f "$AGG_SEG_SCRIPT"

    # --- ident_all: Identification for ALL houses ---
    IDENT_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_ident_${config_name}_XXXXXX.sh")
    cat > "$IDENT_SCRIPT" << IDENT_EOF
#!/bin/bash
#SBATCH --job-name=ident_${config_name}
#SBATCH --output=${LOG_DIR}/ident_${config_name}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/ident_${config_name}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${AGG_SEG_JOB}

echo "========================================"
echo "IDENTIFICATION ALL HOUSES: ${config_name}"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/identification_analysis"
python scripts/run_identification_report.py \\
    --experiment ${exp_output} \\
    --output-dir ${exp_reports} \\
    --publish identification \\
    --aggregate-only \\
    2>&1

echo "Exit: \$? — End: \$(date)"
IDENT_EOF
    IDENT_JOB=$(sbatch "$IDENT_SCRIPT" | awk '{print $4}')
    rm -f "$IDENT_SCRIPT"

    ALL_IDENT_JOBS+=($IDENT_JOB)
    echo "  ${config_name}: agg_pre=${AGG_PRE_JOB} -> agg_seg=${AGG_SEG_JOB} -> ident=${IDENT_JOB}"
done
echo ""

# ============================================================
# PHASE 3: Cross-experiment comparison
# Runs AFTER all identification jobs complete.
# Scans ALL experiments in OUTPUT/experiments/ and produces
# a unified comparison CSV + HTML report.
# ============================================================
IDENT_DEPS=$(IFS=:; echo "${ALL_IDENT_JOBS[*]}")

COMPARE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_compare_XXXXXX.sh")
cat > "$COMPARE_SCRIPT" << COMPARE_EOF
#!/bin/bash
#SBATCH --job-name=compare_all
#SBATCH --output=${LOG_DIR}/compare_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/compare_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${IDENT_DEPS}

echo "========================================"
echo "CROSS-EXPERIMENT COMPARISON"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"

echo "Scanning experiments in ${ALL_EXPERIMENTS_DIR}..."
python scripts/compare_experiments.py \\
    --scan \\
    --scan-dir ${ALL_EXPERIMENTS_DIR} \\
    --output-dir ${ALL_EXPERIMENTS_DIR}/../comparisons \\
    2>&1

echo "Exit: \$? — End: \$(date)"
COMPARE_EOF

COMPARE_JOB=$(sbatch "$COMPARE_SCRIPT" | awk '{print $4}')
rm -f "$COMPARE_SCRIPT"

echo "Phase 3: Comparison job ${COMPARE_JOB} (afterany all identification jobs)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "All jobs submitted!"
echo "============================================================"
echo ""
echo "Experiments:"
for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config_name norm_method <<< "$exp_entry"
    echo "  ${config_name} (${norm_method}) -> ${EXP_OUTPUTS[$config_name]}"
done
echo ""
echo "SLURM jobs:"
echo "  Phase 1: Array ${ARRAY_JOB_ID} (${TASK_COUNT} tasks, %${MAX_CONCURRENT})"
echo "  Phase 2: ${#ALL_IDENT_JOBS[@]} aggregate chains (3 jobs each)"
echo "  Phase 3: Comparison ${COMPARE_JOB}"
echo ""
echo "Manifest:    ${MANIFEST}"
echo "Comparison:  ${ALL_EXPERIMENTS_DIR}/../comparisons/"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Progress: squeue -u \$USER -t RUNNING | wc -l"
echo "============================================================"
