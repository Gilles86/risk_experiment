#!/bin/bash
#SBATCH --job-name=correct_freesurfer
#SBATCH --output=/home/cluster/gdehol/logs/correct_freesurfer_%A-%a.txt
#SBATCH --partition=generic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
export PARTICIPANT_LABEL=$(printf "sub-%02d" $SLURM_ARRAY_TASK_ID)
export FREESURFER_HOME=/data/gdehol/freesurfer
export SUBJECTS_DIR=/scratch/gdehol/ds-risk/derivatives/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
recon-all -s $PARTICIPANT_LABEL -autorecon2 -threads 31
recon-all -s $PARTICIPANT_LABEL -autorecon3 -threads 31
