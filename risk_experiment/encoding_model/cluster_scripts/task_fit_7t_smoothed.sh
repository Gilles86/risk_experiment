#!/bin/bash
#SBATCH --job-name=fit_nprf_unsmoothed
#SBATCH --output=/home/cluster/gdehol/logs/task_fit_7t_smoothed_%A-%a.txt
#SBATCH --partition=vesta
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=10:00
module load volta
module load nvidia/cuda11.2-cudnn8.1.0

. $HOME/init_conda.sh
. $HOME/init_freesurfer.sh
#. $HOME/bashrc.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/risk_experiment/risk_experiment/encoding_model/fit_task.py $PARTICIPANT_LABEL 7t2 --bids_folder /scratch/gdehol/ds-risk --denoise --retroicor --smoothed