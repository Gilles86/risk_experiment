#!/usr/bin/env python

import os
import os.path as op
from datetime import datetime
from itertools import product

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    
job_directory = "%s/.job" %os.getcwd()

if not op.exists(job_directory):
    os.makedirs(job_directory)


sourcedata = '/scratch/gdehol/ds-risk'

subjects = [f'{subject:02d}' for subject in range(2, 33)]
subjects.pop(subjects.index('24'))

sessions = ['3t2', '7t2']
masks = ['npc', 'wang15ipsL', 'wang15ipsR', 'wang15ips']

n_voxels = [100, 250]
smoothed = [False]
pca_confounds = [False]
denoise = [True]
retroicor = [True]

for ix, (subject, session, mask, nv, smooth, pcac, dn, retroi) in enumerate(product(subjects, sessions, masks, n_voxels, smoothed, pca_confounds, denoise, retroicor)):
# for ix, (nv, subject, session, mask) in enumerate(missing):
    print(f'*** RUNNING {subject}, {mask}, {nv} {smooth} {pcac}')

    job_file = os.path.join(job_directory, f"{ix}.job")
    
    now = datetime.now()
    time_str = now.strftime("%Y.%m.%d_%H.%M.%S")

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        id = f'{subject}.{session}.{mask}.{nv}.{time_str}'
        fh.writelines(f"#SBATCH --job-name=decode_volume.{id}.job\n")
        fh.writelines(f"#SBATCH --output={os.environ['HOME']}/.out/decode_volume.{id}.txt\n")
        fh.writelines("#SBATCH --partition=vesta\n")
        fh.writelines("#SBATCH --time=30:00\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --mem=96G\n")
        fh.writelines("#SBATCH --gres gpu:1\n")
        fh.writelines("module load volta\n")
        fh.writelines("module load nvidia/cuda11.2-cudnn8.1.0\n")
        fh.writelines(". $HOME/init_conda.sh\n")
        fh.writelines(". $HOME/init_freesurfer.sh\n")
        fh.writelines("conda activate tf2-gpu\n")
        # cmd = f"python $HOME/git/risk_experiment/risk_experiment/encoding_model/decode.py {subject} {session} --bids_folder /scratch/gdehol/ds-risk --n_voxels {nv} --mask {mask}"
        cmd = f"python $HOME/git/risk_experiment/risk_experiment/encoding_model/decode.py {subject} {session} --bids_folder /scratch/gdehol/ds-risk --n_voxels {nv} --mask {mask}"

        if smooth:
            cmd += ' --smoothed'

        if pcac:
            cmd += ' --pca_confounds'

        if dn:
            cmd += ' --denoise'

        if retroi:
            cmd += ' --retroicor'

        fh.writelines(cmd)
        print(cmd)

    os.system("sbatch %s" %job_file)
