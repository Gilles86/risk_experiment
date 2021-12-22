#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --output=/home/cluster/gdehol/logs/res_fmriprep_%A-%a.txt
#SBATCH --partition=generic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
export SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow
singularity run -B /scratch/gdehol/templateflow:/opt/templateflow --cleanenv /data/gdehol/containers/fmriprep-20.2.2.simg /scratch/gdehol/ds-risk /scratch/gdehol/ds-risk/derivatives participant --participant-label $PARTICIPANT_LABEL  --output-spaces MNI152NLin2009cAsym T1w fsaverage fsnative  --dummy-scans 3 --skip_bids_validation -w /scratch/gdehol/workflow_folders --no-submm-recon
