"""
Run disaggregation for a single month through all threshold iterations.

Enables month-level parallelism on HPC clusters: each month runs
detection -> matching -> segmentation for all 4 iterations independently.
After all months complete, run identification separately.

Usage:
    # List available months for a house:
    python scripts/run_single_month.py --house_id 305 --list_months

    # Run by month index (auto-discovered from input directory):
    python scripts/run_single_month.py --house_id 305 --month_index 0

    # Run by explicit month:
    python scripts/run_single_month.py --house_id 305 --month 07_2021

    # SBATCH array (auto-discover months):
    #   N_MONTHS=$(python scripts/run_single_month.py --house_id 305 --list_months | wc -l)
    #   sbatch --array=0-$((N_MONTHS-1)) run_months.sh

After all months complete, run identification:
    python scripts/run_identification.py --experiment_dir OUTPUT/experiments/exp010_XXX --house_id 305
"""
import sys
import os
import re
import time
import logging
import importlib
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
_SCRIPT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(_SCRIPT_DIR / "src"))


def _discover_months(input_path: str, house_id: str) -> list:
    """Discover available months from the house's input directory.

    Scans for files matching {house_id}_MM_YYYY.pkl or .csv and returns
    sorted list of 'MM_YYYY' strings.
    """
    from core.data_loader import find_house_data_path

    data_path = find_house_data_path(input_path, house_id)
    data_path = Path(data_path)

    if not data_path.is_dir():
        # Single file — extract month from data
        return [data_path.stem.split('_', 1)[-1]] if '_' in data_path.stem else []

    # Pattern: {house_id}_MM_YYYY.pkl or .csv
    pattern = re.compile(rf'^{re.escape(house_id)}_(\d{{2}}_\d{{4}})\.(pkl|csv)$')
    months = []
    for f in data_path.iterdir():
        m = pattern.match(f.name)
        if m:
            months.append(m.group(1))

    # Sort chronologically: by year then month
    months.sort(key=lambda s: (int(s.split('_')[1]), int(s.split('_')[0])))
    return months


def main():
    parser = argparse.ArgumentParser(
        description="Run disaggregation for a single month (all iterations)"
    )
    parser.add_argument("--house_id", required=True, type=str,
                        help="House ID to process")

    # Month selection: one of --list_months, --month_index, or --month
    month_group = parser.add_mutually_exclusive_group()
    month_group.add_argument("--list_months", action="store_true",
                             help="List available months and exit (one per line)")
    month_group.add_argument("--month_index", type=int, default=None,
                             help="0-based index into auto-discovered months (for SBATCH --array)")
    month_group.add_argument("--month", type=str, default=None,
                             help="Explicit month, format: MM_YYYY (e.g. 07_2021)")

    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name (default: exp010_dynamic_threshold)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Shared output directory (MUST be same for all months of same experiment)")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Input data directory (default: INPUT/HouseholdData)")
    args = parser.parse_args()

    input_path = args.input_path or str(_SCRIPT_DIR.parent / "INPUT" / "HouseholdData")

    # --- List months mode ---
    if args.list_months:
        months = _discover_months(input_path, args.house_id)
        if not months:
            print(f"No monthly files found for house {args.house_id} in {input_path}", file=sys.stderr)
            sys.exit(1)
        for m in months:
            print(m)
        return

    # --- Resolve month ---
    if args.month_index is not None:
        months = _discover_months(input_path, args.house_id)
        if not months:
            print(f"No monthly files found for house {args.house_id} in {input_path}", file=sys.stderr)
            sys.exit(1)
        if args.month_index < 0 or args.month_index >= len(months):
            print(f"Error: --month_index {args.month_index} out of range (0-{len(months)-1})", file=sys.stderr)
            print(f"Available months: {months}", file=sys.stderr)
            sys.exit(1)
        month = months[args.month_index]
    elif args.month:
        parts = args.month.split('_')
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            print(f"Error: --month must be MM_YYYY format (e.g. 07_2021), got: {args.month}", file=sys.stderr)
            sys.exit(1)
        month = args.month
    else:
        parser.error("One of --list_months, --month_index, or --month is required")
        return

    # --- Load experiment config ---
    from core.config import get_experiment, DEFAULT_EXPERIMENT, save_experiment_metadata
    experiment_name = args.experiment_name or DEFAULT_EXPERIMENT
    exp_config = get_experiment(experiment_name)

    if exp_config.threshold_schedule is None:
        print("Error: Month-level parallelism only supported for dynamic threshold experiments", file=sys.stderr)
        sys.exit(1)

    threshold_schedule = exp_config.threshold_schedule
    output_path = args.output_path or str(
        _SCRIPT_DIR / "OUTPUT" / "experiments" / f"{exp_config.exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/logs", exist_ok=True)

    # Set up logging
    log_file = f"{output_path}/logs/month_{args.house_id}_{month}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Single-month disaggregation: house={args.house_id}, month={month}")
    logger.info(f"Threshold schedule: {threshold_schedule}")
    logger.info(f"Output: {output_path}")

    # Save experiment metadata (idempotent — all months write the same file)
    save_experiment_metadata(exp_config, output_path, git_hash=None)

    # Set up paths and reload modules
    import core.paths
    importlib.reload(core.paths)
    core.paths.OUTPUT_BASE_PATH = output_path
    core.paths.OUTPUT_ROOT = output_path
    core.paths.INPUT_DIRECTORY = output_path
    core.paths.LOGS_DIRECTORY = f"{output_path}/logs/"
    core.paths.RAW_INPUT_DIRECTORY = input_path

    import core.logging_setup
    importlib.reload(core.logging_setup)
    import core as core_module
    importlib.reload(core_module)

    import disaggregation.rectangle.pipeline.detection_step
    import disaggregation.rectangle.pipeline.matching_step
    import disaggregation.rectangle.pipeline.segmentation_step
    importlib.reload(disaggregation.rectangle.pipeline.detection_step)
    importlib.reload(disaggregation.rectangle.pipeline.matching_step)
    importlib.reload(disaggregation.rectangle.pipeline.segmentation_step)

    process_detection = disaggregation.rectangle.pipeline.detection_step.process_detection
    process_matching = disaggregation.rectangle.pipeline.matching_step.process_matching
    process_segmentation = disaggregation.rectangle.pipeline.segmentation_step.process_segmentation

    from core import find_house_data_path, find_previous_run_summarized

    t_start = time.time()

    # Run all iterations for this single month
    for run_number, threshold in enumerate(threshold_schedule):
        logger.info(f"\n{'#'*60}")
        logger.info(f"ITERATION {run_number}: THRESHOLD = {threshold}W, MONTH = {month}")
        logger.info(f"{'#'*60}")

        run_dir = Path(output_path) / f"run_{run_number}"
        output_dir = str(run_dir / f"house_{args.house_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Verify input exists
        try:
            if run_number == 0:
                find_house_data_path(input_path, args.house_id)
            else:
                find_previous_run_summarized(output_path, args.house_id, run_number)
        except FileNotFoundError as e:
            if run_number == 0:
                logger.error(f"Input data not found: {e}")
                sys.exit(1)
            else:
                logger.info(f"No summarized data from previous run for this month, stopping")
                break

        try:
            t0 = time.time()
            process_detection(
                house_id=args.house_id, run_number=run_number,
                threshold=threshold, config=exp_config,
                month_filter=month,
            )
            logger.info(f"  Detection: {time.time() - t0:.1f}s")

            t0 = time.time()
            process_matching(
                house_id=args.house_id, run_number=run_number,
                threshold=threshold, month_filter=month, config=exp_config,
            )
            logger.info(f"  Matching: {time.time() - t0:.1f}s")

            t0 = time.time()
            process_segmentation(
                house_id=args.house_id, run_number=run_number,
                skip_large_file=True,
                use_nan_imputation=getattr(exp_config, 'use_nan_imputation', False),
                month_filter=month,
            )
            logger.info(f"  Segmentation: {time.time() - t0:.1f}s")

        except Exception as e:
            logger.error(f"Error in iteration {run_number}: {e}", exc_info=True)
            sys.exit(1)

        # Check if this month produced summarized output for next iteration
        summarized_file = run_dir / f"house_{args.house_id}" / "summarized" / f"summarized_{args.house_id}_{month}.pkl"
        if not summarized_file.exists():
            logger.info(f"No summarized output for {month}, stopping iterations")
            break

    elapsed = time.time() - t_start
    logger.info(f"\nDone: house={args.house_id}, month={month}, {elapsed:.1f}s total")
    print(f"\nMonth {month} complete ({elapsed:.1f}s). Run identification after all months finish:")
    print(f"  python scripts/run_identification.py --experiment_dir {output_path} --house_id {args.house_id}")


if __name__ == '__main__':
    main()


# =============================================================================
# SBATCH example — auto-discovers months from input directory:
#
# #!/bin/bash
# #SBATCH --job-name=disagg_305
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=4G
# #SBATCH --output=logs/month_%A_%a.out
#
# # First, discover how many months exist:
# #   N=$(cd experiment_pipeline && python scripts/run_single_month.py --house_id 305 --list_months | wc -l)
# #   sbatch --array=0-$((N-1)) this_script.sh
#
# OUTPUT_PATH="OUTPUT/experiments/exp010_parallel"
#
# cd experiment_pipeline
# python scripts/run_single_month.py \
#     --house_id 305 \
#     --month_index $SLURM_ARRAY_TASK_ID \
#     --output_path $OUTPUT_PATH
#
# # After all array tasks complete, submit identification:
# # sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID run_identification.sh
# =============================================================================
