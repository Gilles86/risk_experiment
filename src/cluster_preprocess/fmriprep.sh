#!/bin/bash
#
#SBATCH --job-name=fmriprep
#SBATCH --output=/home/gholland/logs/res_fmriprep_%A-%a.txt
#
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
singularity run $HOME/fmriprep-20.1.1.simg /home/gholland/data/numerosity_7t/ds-marcus /home/gholland/data/numerosity_7t/ds-marcus/derivatives participant --participant-label marcus --fs-no-reconall -w /scratch  --output-spaces MNI152NLin2009cAsym T1w  --skip_bids_validation
