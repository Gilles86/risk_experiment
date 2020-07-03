import glob
import pandas as pd
import os
import os.path as op
import argparse
import re
import shutil
from warnings import warn
import json

def main(sourcedata, session='3t'):
    reg = re.compile('(?P<path>.+)/sub-(?P<subject>.+)')

    match = reg.match(sourcedata)
    ds_folder = op.dirname(match.group(1))
    subject = match.group(2)
    print(subject)
    subject_folder = op.join(ds_folder, f'sub-{subject}', f'ses-{session}')

    if op.exists(subject_folder):
        raise Exception('Subject folder already exists?')
    else:
        for modality in ['anat', 'fmap', 'eye', 'func']:
            os.makedirs(op.join(subject_folder, modality))

    
    nii_files = glob.glob(op.join(sourcedata, '*.nii'))

    nii_reg = re.compile('.*/sn_[0-9]{8}_[0-9]{6}_(?P<acq>[0-9]+)_1_wip(?P<label>.+).nii')#(?P<name>.+)_?\.nii')

    df = []
    for fn in nii_files:
        if nii_reg.match(fn):
            df.append(nii_reg.match(fn).groupdict())
            df[-1]['fn'] = fn

    df = pd.DataFrame(df).set_index(['label'])
    print(df)
    # Get anatomical data

    if 't1w_tse_ori_' in df.index:
        shutil.copy(df.loc['t1w_tse_ori_'].fn,
                op.join(subject_folder, 'anat', f'sub-{subject}_ses-{session}_TSE.nii'))
    else:
        warning('TSE missing!')

    if 't1w3danat_4_' in df.index:
        shutil.copy(df.loc['t1w3danat_4_'].fn,
                op.join(subject_folder, 'anat', f'sub-{subject}_ses-{session}_T1w.nii'))
    else:
        warning('T1w missing!')

    if 't2w_1mm_spl' in df.index:
        shutil.copy(df.loc['t2w_1mm_spl'].fn,
                op.join(subject_folder, 'anat', f'sub-{subject}_ses-{session}_T2starw.nii'))
    else:
        warning('T2starw missing!')

    sidecar_json = {
	"MagneticFieldStrength": 3,
	"ParallelReductionFactorInPlane": 2,
	"RepetitionTime": 2.506,
        "TotalReadoutTime": 0.04,
        "TaskName":"Numerosity mapper" }
    
    for mapper_run in range(1, 5):
        if f'mapper_run{mapper_run}' in df.index:
            if mapper_run % 2 == 0:
                sidecar_json['PhaseEncodingDirection'] = 'i-'
            else:
                sidecar_json['PhaseEncodingDirection'] = 'i'

            with open(op.join(subject_folder, 'func', f'sub-{subject}_ses-{session}_task-mapper_run-{mapper_run}_bold.json'), 'w') as fp:
                    json.dump(sidecar_json, fp)

            shutil.copy(df.loc[f'mapper_run{mapper_run}'].fn,
                    op.join(subject_folder, 'func', f'sub-{subject}_ses-{session}_task-mapper_run-{mapper_run}_bold.nii'))

    sidecar_json = {"TotalReadoutTime": 0.04}

    for topup_run in range(1, 5):

        if f'mapper_run{topup_run}_' in df.index:
            if topup_run % 2 == 0:
                sidecar_json['PhaseEncodingDirection'] = 'i'
                acq = 'lr'
            else:
                sidecar_json['PhaseEncodingDirection'] = 'i-'
                acq = 'rl'
            
            sidecar_json['IntendedFor'] = f'ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{topup_run}_bold.nii'

            with open(op.join(subject_folder, 'fmap', f'sub-{subject}_ses-{session}_run-{topup_run}_epi.json'), 'w') as fp:
                    json.dump(sidecar_json, fp)

            shutil.copy(df.loc[f'mapper_run{topup_run}_'].fn,
                    op.join(subject_folder, 'fmap', f'sub-{subject}_ses-{session}_run-{topup_run}_epi.nii'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sourcedata', default=None)
    args = parser.parse_args()

    main(args.sourcedata)
