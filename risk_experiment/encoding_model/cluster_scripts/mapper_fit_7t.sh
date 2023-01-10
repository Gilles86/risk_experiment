#!/bin/bash
#SBATCH --job-name=fit_mapper_7t_unsmoothed
#SBATCH --output=/home/gdehol/logs/mapper_fit_7t_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=20:00

source /etc/profile.d/lmod.sh
module load gpu
module load cuda

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/risk_experiment/risk_experiment/encoding_model/fit_mapper.py $PARTICIPANT_LABEL 7t1 --bids_folder /scratch/gdehol/ds-risk --natural_space
