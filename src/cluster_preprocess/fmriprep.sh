#!/bin/bash
#
#SBATCH --job-name=fmriprep
#SBATCH --output=/home/gholland/logs/res_fmriprep_%A-%a.txt
#
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
export SINGULARITYENV_FS_LICENSE=$FREESURFER_HOME/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
singularity run --cleanenv $HOME/fmriprep-20.2.1.simg /home/gholland/data/numerosity_7t/ds-numrisk /home/gholland/data/numerosity_7t/ds-numrisk/derivatives participant --participant-label $PARTICIPANT_LABEL -w /scratch  --output-spaces MNI152NLin2009cAsym T1w fsaverage fsnative  --skip_bids_validation --no-submm-recon --dummy-scans 3
