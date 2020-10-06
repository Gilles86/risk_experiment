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


def main(subject, session, bids_folder, sourcedata=None, overwrite=True):

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

    scanner = session[:2]
    fieldstrength = int(session[0])


    create_bids_folder(bids_folder, ['anat', 'func', 'fmap'])

    df = []

    if fieldstrength == 7:
        nii_reg = re.compile(
            f'.*/ri_[0-9]+_[0-9]+_(?P<acq>[0-9]+)_[0-9]+_ses-{session}_(?P<label>(task-(?P<task>[a-z]+)_)?(run-(?P<run>[0-9]+)_)?(?P<suffix>.+))V4.nii')
    else:
        nii_reg = re.compile(
            f'.*/sn_[0-9]+_[0-9]+_(?P<acq>[0-9]+)_[0-9]+_ses-{session}_(?P<label>(task-(?P<task>[a-z]+)_)?(run-(?P<run>[0-9]+)_)?(?P<suffix>.+))_spli(ts)?.nii')


    df = []
    for fn in nii_files:
        # print(nii_reg)
        # print(fn)
        if nii_reg.match(fn):
            print(fn)
            df.append(nii_reg.match(fn).groupdict())
            df[-1]['fn'] = fn

    df = pd.DataFrame(df).set_index(['suffix'])
    print(df)

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

        with open(op.join(bids_folder, 'func', f'sub-{subject}_ses-{session}_{row.label}.json'), 'w') as fp:
            json.dump(sidecar_json, fp)

        shutil.copy(row.fn,
                    op.join(bids_folder, 'func', f'sub-{subject}_ses-{session}_{row.label}.nii'))

    runs = [run for run in sorted(df.loc['bold'].run.unique())]
    tmp = df.loc['bold'].set_index('run')

    sidecar_json = {"TotalReadoutTime": 0.04}

    # Make topups
    for i, topup_run in enumerate(runs[1:]):
        print(runs[i], topup_run)
        intended_for_run = runs[i]
        if (int(topup_run) % 2) == (int(intended_for_run) %2):
            topup_run = str(int(topup_run) + 1)

        topup = image.load_img(tmp.loc[topup_run].fn)
        topup = image.index_img(topup, range(10))

        if int(topup_run) % 2 == 0:
            sidecar_json['PhaseEncodingDirection'] = 'i-'
        else:
            sidecar_json['PhaseEncodingDirection'] = 'i'

        intended_for_bold = op.join('func', f'sub-{subject}_ses-{session}_{tmp.loc[intended_for_run].label}.nii')

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

    intended_for_bold = op.join(f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_{tmp.loc[intended_for_run].label}.nii')

    sidecar_json['IntendedFor'] = intended_for_bold

    fn = op.join(bids_folder, 'fmap', f'sub-{subject}_ses-{session}_run-{intended_for_run}_epi')

    with open(fn + '.json', 'w') as fp:
        json.dump(sidecar_json, fp)

    topup.to_filename(fn + '.nii')


    tmp = df.drop('bold')
    
    for suffix, row in tmp.iterrows():
        shutil.copy(row.fn,
                    op.join(bids_folder, 'anat', f'sub-{subject}_ses-{session}_{row.label}.nii'))



    if scanner not in ['3t', '7t']:
        raise Exception('Session should start with 3t or 7t')
        # nii_reg = re.compile('.*/sn_[0-9]{8}_[0-9]{6}_(?p<acq>[0-9]+)_1_wip(?p<label>.+).nii')#(?p<name>.+)_?\.nii')

        # df = []
        # for fn in nii_files:
        # if nii_reg.match(fn):
        # df.append(nii_reg.match(fn).groupdict())
        # df[-1]['fn'] = fn

        # df = pd.DataFrame(df).set_index(['label'])
        # print(df)
        # # Get anatomical data

        # if 't1w_tse_ori_' in df.index:
        # shutil.copy(df.loc['t1w_tse_ori_'].fn,
        # op.join(subject_folder, 'anat', f'sub-{subject}_ses-{session}_TSE.nii'))
        # else:
        # warning('TSE missing!')

        # if 't1w3danat_4_' in df.index:
        # shutil.copy(df.loc['t1w3danat_4_'].fn,
        # op.join(subject_folder, 'anat', f'sub-{subject}_ses-{session}_T1w.nii'))
        # else:
        # warning('T1w missing!')

        # if 't2w_1mm_spl' in df.index:
        # shutil.copy(df.loc['t2w_1mm_spl'].fn,
        # op.join(subject_folder, 'anat', f'sub-{subject}_ses-{session}_T2starw.nii'))
        # else:
        # warning('T2starw missing!')

        # sidecar_json = {
        # "MagneticFieldStrength": 3,
        # "ParallelReductionFactorInPlane": 2,
        # "RepetitionTime": 2.506,
        # "TotalReadoutTime": 0.04,
        # "TaskName":"Numerosity mapper" }

        # for mapper_run in range(1, 5):
        # if f'mapper_run{mapper_run}' in df.index:
        # if mapper_run % 2 == 0:
        # sidecar_json['PhaseEncodingDirection'] = 'i-'
        # else:
        # sidecar_json['PhaseEncodingDirection'] = 'i'

        # with open(op.join(subject_folder, 'func', f'sub-{subject}_ses-{session}_task-mapper_run-{mapper_run}_bold.json'), 'w') as fp:
        # json.dump(sidecar_json, fp)

        # shutil.copy(df.loc[f'mapper_run{mapper_run}'].fn,
        # op.join(subject_folder, 'func', f'sub-{subject}_ses-{session}_task-mapper_run-{mapper_run}_bold.nii'))

        # sidecar_json = {"TotalReadoutTime": 0.04}

        # for topup_run in range(1, 5):

        # if f'mapper_run{topup_run}_' in df.index:
        # if topup_run % 2 == 0:
        # sidecar_json['PhaseEncodingDirection'] = 'i'
        # acq = 'lr'
        # else:
        # sidecar_json['PhaseEncodingDirection'] = 'i-'
        # acq = 'rl'

        # sidecar_json['IntendedFor'] = f'ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{topup_run}_bold.nii'

        # with open(op.join(subject_folder, 'fmap', f'sub-{subject}_ses-{session}_run-{topup_run}_epi.json'), 'w') as fp:
        # json.dump(sidecar_json, fp)

        # shutil.copy(df.loc[f'mapper_run{topup_run}_'].fn,
        # op.join(subject_folder, 'fmap', f'sub-{subject}_ses-{session}_run-{topup_run}_epi.nii'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default=op.join(os.environ['HOME'], 'Science', 'numerosity_7t', 'data'))
    parser.add_argument('--sourcedata', default=None)
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder, args.sourcedata)
