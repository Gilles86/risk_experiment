#!/bin/bash
#SBATCH --job-name=fit_mapper_3t_unsmoothed
#SBATCH --output=/home/gdehol/logs/mapper_fit_3t_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=10:00

source /etc/profile.d/lmod.sh
module load cuda
module load gpu

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/risk_experiment/risk_experiment/encoding_model/fit_mapper.py $PARTICIPANT_LABEL 3t1 --bids_folder /scratch/gdehol/ds-risk --natural_space
