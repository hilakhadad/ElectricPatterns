#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name house_analysis
#SBATCH --output /home/hilakese/role_based_segregation_dev/house_analysis/logs/house_analysis-%J.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

### Print some data to output file ###
echo "Starting house analysis"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Date: $(date)"

### Start your code below ####
module load anaconda
source activate nilm_stat_env

cd /home/hilakese/role_based_segregation_dev/house_analysis/scripts/

# Run analysis - uses default paths automatically
# -u flag for unbuffered output (shows progress immediately)
python -u run_analysis.py

echo "Analysis completed at $(date)"
