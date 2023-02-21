#!/bin/bash
#SBATCH --job-name=decode_select_voxels_cv_3
#SBATCH --output=/home/cluster/gdehol/logs/decode_select_voxels_cv_3t_%A_%a.txt
#SBATCH --partition=generic
#SBATCH --ntasks=1
#SBATCH -c16
#SBATCH --time=1:00:00

. $HOME/init_conda.sh
. $HOME/init_freesurfer.sh
#. $HOME/bashrc.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/risk_experiment/risk_experiment/encoding_model/decode_select_voxels_cv.py $PARTICIPANT_LABEL 3t2 --bids_folder /scratch/gdehol/ds-risk --denoise --retroicor --mask npcr
