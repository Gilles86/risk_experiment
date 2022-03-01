#!/bin/bash
#SBATCH --job-name=fit_nprf_surf
#SBATCH --output=/home/cluster/gdehol/logs/fit_nprf_surf_%A-%a.txt
#SBATCH --partition=volta
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=2:00:00
module load volta
module load nvidia/cuda11.2-cudnn8.1.0

. $HOME/init_conda.sh
. $HOME/init_freesurfer.sh
#. $HOME/bashrc.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/risk_experiment/risk_experiment/preproc/smooth_surf.py $PARTICIPANT_LABEL 3t1 --bids_folder /scratch/gdehol/ds-risk
python $HOME/git/risk_experiment/risk_experiment/preproc/smooth_surf.py $PARTICIPANT_LABEL 7t1 --bids_folder /scratch/gdehol/ds-risk

python $HOME/git/risk_experiment/risk_experiment/encoding_model/fit_mapper_surf.py $PARTICIPANT_LABEL 3t1 --bids_folder /scratch/gdehol/ds-risk --smoothed
python $HOME/git/risk_experiment/risk_experiment/encoding_model/fit_mapper_surf.py $PARTICIPANT_LABEL 7t1 --bids_folder /scratch/gdehol/ds-risk --smoothed
