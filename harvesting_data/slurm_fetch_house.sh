#!/bin/bash
#SBATCH --job-name=harvest
#SBATCH --output=/home/hilakese/role_based_segregation_dev/harvesting_data/logs/harvest_%A_%a.log
##SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

##
## SLURM job array script for fetching house data.
## Submit with: sbatch --array=1-204 slurm_fetch_house.sh
##
## The array index corresponds to line number in the token file (1-indexed, after header).
## To run specific houses: sbatch --array=1,5,10 slurm_fetch_house.sh
##

set -e

## Redirect stderr to stdout so both go to same log file
exec 2>&1

## Check if running as array job
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "Error: Must run as array job. Use: sbatch --array=1-204 slurm_fetch_house.sh"
    exit 1
fi

## Configuration - absolute paths for cluster
PROJECT_DIR="/home/hilakese/role_based_segregation_dev"
TOKEN_FILE="${PROJECT_DIR}/INPUT/id_token.csv"
OUTPUT_DIR="${PROJECT_DIR}/INPUT/UpdatatedHouseData"

## Get house ID from array index (line number = array index + 1 for header)
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
HOUSE_ID=$(sed -n "${LINE_NUM}p" "$TOKEN_FILE" | cut -d',' -f1)

if [[ -z "$HOUSE_ID" || "$HOUSE_ID" == "ID" ]]; then
    echo "Error: Invalid house ID at line $LINE_NUM: '$HOUSE_ID'"
    exit 1
fi

## Update job name to include house ID (short format for squeue display)
scontrol update JobId=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} Name="h_${HOUSE_ID}"

echo "=========================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "House ID: $HOUSE_ID"
echo "=========================================="

cd "$PROJECT_DIR"

## Activate conda environment if needed
## source ~/.bashrc
## conda activate your_env

python -m harvesting_data.fetch_single_house \
    --house "$HOUSE_ID" \
    --token-file "$TOKEN_FILE" \
    --output "$OUTPUT_DIR"

echo "Done!"
