#!/bin/sh
export PARTICIPANT_LABEL=$(printf "%02d" $1)
echo $PARTICIPANT_LABEL
fmriprep-docker /data/ds-risk /data/ds-risk/derivative participant --participant-label $PARTICIPANT_LABEL -w /data/tmp  --output-spaces MNI152NLin2009cAsym T1w fsaverage fsnative  --skip_bids_validation --no-submm-recon --dummy-scans 3
