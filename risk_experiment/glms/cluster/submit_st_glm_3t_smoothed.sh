#!/bin/bash
#SBATCH --job-name=fit_st_denoise
#SBATCH --output=/home/cluster/gdehol/logs/fit_st_denoise_%A-%a.txt
#SBATCH --partition=generic
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=60:00

. $HOME/.bashrc

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

python $HOME/git/risk_experiment/risk_experiment/glms/fit_single_trials_volume_denoise.py $PARTICIPANT_LABEL 3t2 --bids_folder /scratch/gdehol/ds-risk --smoothed --retroicor
