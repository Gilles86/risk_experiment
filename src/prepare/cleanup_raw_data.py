import glob
import pandas as pd
import os
import os.path as op
import argparse
import re
import shutil
from warnings import warn
import json
from nilearn import image
import numpy as np
from itertools import product

def main(subject, session, bids_folder, sourcedata=None, overwrite=True, physiology_only=False):

    if sourcedata is None:
        sourcedata = op.join(bids_folder, 'sourcedata')

    bids_folder = op.join(bids_folder, f'sub-{subject}', f'ses-{session}')

    sourcedata_mri = op.join(
        sourcedata, f'sub-{subject}', 'mri', f'ses-{session}')
    sourcedata_behavior = op.join(
        sourcedata, f'sub-{subject}', 'behavior', f'ses-{session}')

    nii_files = glob.glob(op.join(sourcedata_mri, '*.nii'))

    def create_bids_folder(bids_folder, modalities):
        if (not overwrite) & op.exists(bids_folder):
            raise Exception('Subject folder already exists?')
        else:
            for modality in modalities:
                mod_dir = op.join(bids_folder, modality)
                if not op.exists(mod_dir):
                    os.makedirs(mod_dir)

    fieldstrength = int(session[0])

    create_bids_folder(bids_folder, ['anat', 'func', 'fmap'])

    if not physiology_only:
        if fieldstrength == 7:
            nii_reg = re.compile(
                f'.*/ri_[0-9]+_[0-9]+_(?P<acq>[0-9]+)_[0-9]+_ses-{session}(?P<label>(_task-(?P<task>calibration|mapper|task))?(_run-(?P<run>[0-9]+))?(_(?P<suffix>.+))?)V4.nii')
        else:
            nii_reg = re.compile(
                f'.*/sn_[0-9]+_[0-9]+_(?P<acq>[0-9]+)_[0-9]+_ses{session}_(?P<label>.+)\.nii')



        df = []
        for fn in nii_files:
            if nii_reg.match(fn):
                df.append(nii_reg.match(fn).groupdict())
                df[-1]['fn'] = fn

        df = pd.DataFrame(df)

        if 'suffix' not in df.columns:
            df['suffix'] = np.nan

        df.loc[df.suffix.isnull(), 'suffix'] = 'bold'


        mapper = {'t1w':'T1w', 'tse':'TSE', 't2starw':'T2starw'}
        df['suffix'] = df['suffix'].map(lambda suffix: mapper[suffix] if suffix in mapper else suffix)
        df['label'] = df['label'].apply(lambda x: x[1:] if x[0] == '_' else x)
        df = df.set_index('suffix')
        df = df.sort_values('acq')

        if 'run' not in df.columns:
            bold = df.loc['bold']
            df.loc['bold', 'run'] = np.arange(1, len(bold)+1)

        if 'task' not in df.columns:
            bold = df.loc['bold']
            task = 'mapper' if session == '3t1' else 'task'
            df.loc['bold', 'task'] = task

            df.loc['bold', 'label'] = df.loc['bold'].apply(lambda row: f'task-{row.task}_run-{row.run}', 1)

        if ('T1w' in df.index) and (len(df.loc['T1w']) > 1):
            df.loc['T1w', 'run'] = range(1, len(df.loc['T1w']) + 1)

        if fieldstrength == 7:
            sidecar_json = {
                "MagneticFieldStrength": 7,
                "ParallelReductionFactorInPlane": 3,
                "RepetitionTime": 2.3,
                "TotalReadoutTime": 0.04,
                "TaskName": "Numerosity mapper"}
        else:
            sidecar_json = {
                "MagneticFieldStrength": 3,
                "ParallelReductionFactorInPlane": 2,
                "RepetitionTime": 2.3,
                "TotalReadoutTime": 0.04,
                "TaskName": "Numerosity mapper"}

        for ix, row in df.loc['bold'].iterrows():
            if int(row.run) % 2 == 0:
                sidecar_json['PhaseEncodingDirection'] = 'i-'
            else:
                sidecar_json['PhaseEncodingDirection'] = 'i'
            sidecar_json['task'] = row.task

            with open(op.join(bids_folder, 'func', f'sub-{subject}_ses-{session}_{row.label}_bold.json'), 'w') as fp:
                json.dump(sidecar_json, fp)

            shutil.copy(row.fn,
                        op.join(bids_folder, 'func', f'sub-{subject}_ses-{session}_{row.label}_bold.nii'))

        runs = [run for run in sorted(df.loc['bold'].run.unique())]
        tmp = df.loc['bold'].set_index('run')

        sidecar_json = {"TotalReadoutTime": 0.04}

        # Make topups
        for i, topup_run in enumerate(runs[1:]):
            intended_for_run = runs[i]
            if (int(topup_run) % 2) == (int(intended_for_run) %2):
                topup_run = str(int(topup_run) + 1)

            topup = image.load_img(tmp.loc[topup_run].fn)
            topup = image.index_img(topup, range(10))

            if int(topup_run) % 2 == 0:
                sidecar_json['PhaseEncodingDirection'] = 'i-'
            else:
                sidecar_json['PhaseEncodingDirection'] = 'i'

            intended_for_bold = op.join('func', f'sub-{subject}_ses-{session}_{tmp.loc[intended_for_run].label}_bold.nii')

            sidecar_json['IntendedFor'] = intended_for_bold

            fn = op.join(bids_folder, 'fmap', f'sub-{subject}_ses-{session}_run-{intended_for_run}_epi')

            with open(fn + '.json', 'w') as fp:
                json.dump(sidecar_json, fp)

            topup.to_filename(fn + '.nii')


        # Use before-last run as topup for final run
        intended_for_run = runs[-1]
        topup_run = runs[-2]
        topup = image.load_img(tmp.loc[topup_run].fn)
        topup = image.index_img(topup, range(-10, 0))

        if int(topup_run) % 2 == 0:
            sidecar_json['PhaseEncodingDirection'] = 'i-'
        else:
            sidecar_json['PhaseEncodingDirection'] = 'i'

        intended_for_bold = op.join(f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_{tmp.loc[intended_for_run].label}_bold.nii')

        sidecar_json['IntendedFor'] = intended_for_bold

        fn = op.join(bids_folder, 'fmap', f'sub-{subject}_ses-{session}_run-{intended_for_run}_epi')

        with open(fn + '.json', 'w') as fp:
            json.dump(sidecar_json, fp)

        topup.to_filename(fn + '.nii')


        tmp = df.drop('bold')
        
        for suffix, row in tmp.iterrows():

            # T2star-weighted
            if suffix == 'T2starw':
                im = image.load_img(row.fn) 
                if im.shape[-1] == 1:
                    shutil.copy(row.fn,
                                op.join(bids_folder, 'anat', f'sub-{subject}_ses-{session}_{row.name}.nii'))
                elif im.shape[-1] == 6:
                    for i, (part, echo) in enumerate(product(['mag', 'phase'], [1,2,3])):
                        im_ = image.index_img(im, i)
                        im_.to_filename(op.join(bids_folder, 'anat', f'sub-{subject}_ses-{session}_echo-{echo}_part-{part}_T2starw.nii'))

                    mean_t2starw = image.mean_img(image.index_img(im, [0,1,2]))
                    mean_t2starw.to_filename(op.join(bids_folder, 'anat', f'sub-{subject}_ses-{session}_acq-average_T2starw.nii'))

                elif (im.shape[-1] == 3):
                    for i, echo in enumerate(range(1, 4)):
                        im_ = image.index_img(im, i)
                        im_.to_filename(op.join(bids_folder, 'anat', f'sub-{subject}_ses-{session}_echo-{echo}_part-{part}_T2starw.nii'))

                    mean_t2starw.to_filename(op.join(bids_folder, 'anat', f'sub-{subject}_ses-{session}_acq-average_T2starw.nii'))

            # Everything else anatomical
            else:
                if row.run is None:
                    shutil.copy(row.fn,
                                op.join(bids_folder, 'anat', f'sub-{subject}_ses-{session}_{row.name}.nii'))
                else:
                    shutil.copy(row.fn,
                                op.join(bids_folder, 'anat', f'sub-{subject}_ses-{session}_run-{row.run}_{row.name}.nii'))



    # Physiological files
    log_files = glob.glob(op.join(sourcedata_mri, '*.log'))
    print(log_files)

    if fieldstrength == 7:
        log_reg = re.compile(
            f'.*/SCANPHYSLOG_ri_[0-9]+_[0-9]+_(?P<acq>[0-9]+)_[0-9]+_ses-{session}_task-(?P<task>task|mapper)_run-(?P<run>[0-9]+)V4\.log')
    else:
        log_reg = re.compile(
        f'.*/sn_[0-9]+_[0-9]+_(?P<acq>[0-9]+)_[0-9]+_(?P<task>.+)_run(?P<run>[0-9]+)_spli_scanphyslog.+')

    if session == '3t2':
        log_reg = re.compile(
        f'.*\/sn_[0-9]+_[0-9]+_(?P<acq>[0-9]+)_[0-9]+_ses3t2_tasktas_scanphyslog[0-9]+.log')

        df = []
        for fn in log_files:
            print(fn, log_reg.match(fn))
            if log_reg.match(fn):
                d = log_reg.match(fn).groupdict()
                d['fn'] = fn
                df.append(d)

        df = pd.DataFrame(df).sort_values('acq')
        print(df)
        df['run'] = range(1, len(df)+1)
        df = df.set_index(['run'])

        for run, row in df.iterrows():
            new_fn = op.join(bids_folder, 'func', f'sub-{subject}_ses-{session}_task-task_run-{run}_physio.log')
            print(row.fn, new_fn)
            shutil.copy(row.fn, new_fn)

    else:
        for fn in log_files:
            if log_reg.match(fn):
                d = log_reg.match(fn).groupdict()
                new_fn = op.join(bids_folder, 'func', f'sub-{subject}_ses-{session}_task-{d["task"]}_run-{d["run"]}_physio.log')
                shutil.copy(fn, new_fn)

    if fieldstrength not in [3, 7]:
        raise Exception('Session should start with 3t or 7t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--physiology_only', action='store_true')
    parser.add_argument(
        '--bids_folder', default='/data2/ds-risk/')
    parser.add_argument('--sourcedata', default=None)
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder, args.sourcedata, physiology_only=args.physiology_only)
