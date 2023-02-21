import argparse
import os.path as op
import os
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiMasker
from tqdm import tqdm
import pandas as pd


def main(subject, session, roi, bids_folder, single_trial=False):

    if session.endswith('2'):
        task = 'task'
        if single_trial:
            runs = [1]
        else:
            runs = range(1, 9)
    else:
        if single_trial:
            raise NotImplementedError

        task = 'mapper'
        runs = range(1,5)

    target_dir = op.join(bids_folder, 'derivatives', 'extracted_signal', f'sub-{subject}', f'ses-{session}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)


    mask = op.join(bids_folder, f'derivatives', 'masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-{roi}_mask.nii.gz')

    if single_trial:
        data = op.join(bids_folder, f'derivatives/glm_stim1/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims1_pe.nii.gz')
        data = [data]
    else:
        data = op.join(bids_folder, 'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{run}_space-T1w_desc-preproc_bold.nii.gz')

        data = [data.format(subject=subject, run=run, session=session) for run in runs]

    mask = image.resample_to_img(mask, data[0], interpolation='linear')
    mask = image.math_img('mask > 0.0', mask=mask)

    masker = NiftiMasker(mask) 

    df = []
    keys = []

    for run in tqdm(runs):
        d = pd.DataFrame(masker.fit_transform(data[run-1]))
        d.index.name = 'frame'

        if not single_trial:
            d = (d / d.mean()) * 100 - 100
        
        d = d.mean(1)

        df.append(d)
        if single_trial:
            keys.append((subject, session))
        else:
            keys.append((subject, session, run))

    if single_trial:
        df = pd.concat(df, keys=keys, names=['subject', 'session'])
        df = df.to_frame(roi).droplevel(-1).set_index(pd.Series(np.arange(1, 193), name='trial_nr'), append=True)
        fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{roi}.singletrial_timeseries.tsv')
    else:
        df = pd.concat(df, keys=keys, names=['subject', 'session', 'run']).to_frame(roi)
        fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{roi}_timeseries.tsv')

    df.to_csv(fn, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('session', nargs='?', default=None)
    parser.add_argument('--mask', nargs='?', default='lcL')
    parser.add_argument('--single_trial', action='store_true')
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()
    print(args)

    main(args.subject, args.session, args.mask,  args.bids_folder, single_trial=args.single_trial)
