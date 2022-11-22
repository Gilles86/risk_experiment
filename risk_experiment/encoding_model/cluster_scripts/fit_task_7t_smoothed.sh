#!/bin/bash
#SBATCH --job-name=fit_task_7t_smoothed.sh
#SBATCH --output=/home/cluster/gdehol/logs/fit_task_7t_smoothed_%A_%a.txt
#SBATCH --partition=generic
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH -c16
#SBATCH --time=1:00:00

. $HOME/init_conda.sh
. $HOME/init_freesurfer.sh
#. $HOME/bashrc.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/risk_experiment/risk_experiment/encoding_model/fit_task.py $PARTICIPANT_LABEL 7t2 --bids_folder /scratch/gdehol/ds-risk --denoise --retroicor --smoothed
