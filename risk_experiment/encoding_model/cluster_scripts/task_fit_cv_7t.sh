#!/bin/bash
#SBATCH --job-name=task_fit_7t_cv
#SBATCH --output=/home/gdehol/logs/task_fit_cv_7t_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=1:00:00

source /etc/profile.d/lmod.sh
module load gpu 
module load cuda 

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/risk_experiment/risk_experiment/encoding_model/fit_task_cv.py $PARTICIPANT_LABEL 7t2 --bids_folder /scratch/gdehol/ds-risk --denoise --natural_space
