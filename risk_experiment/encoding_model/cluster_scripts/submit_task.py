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
n_voxels = 250

subjects = [f'{subject:02d}' for subject in range(2, 33)]
sessions = ['3t2', '7t2']
stimulus = ['1']
# masks = ['wang15_ipsL', 'wang15_ipsR', 'npcl', 'npcr']
# masks = ['npc']
# n_voxels = [250]
subjects = ['10']
sessions = ['3t2']


for ix, (subject, session, stim) in enumerate(product(subjects, sessions, stimulus)):
    print(f'*** RUNNING {subject}, {session}, {stim}')

    job_file = os.path.join(job_directory, f"{ix}.job")
    
    now = datetime.now()
    time_str = now.strftime("%Y.%m.%d_%H.%M.%S")

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        id = f'{subject}.{session}.{stim}'
        fh.writelines(f"#SBATCH --job-name=fit_task.{id}.job\n")
        fh.writelines(f"#SBATCH --output={os.environ['HOME']}/.out/fit_task.{id}.txt\n")
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
        cmd = f"python $HOME/git/risk_experiment/risk_experiment/encoding_model/fit_task.py {subject} {session} --bids_folder /scratch/gdehol/ds-risk --stimulus {stim}"
        fh.writelines(cmd)
        print(cmd)

    os.system("sbatch %s" %job_file)
