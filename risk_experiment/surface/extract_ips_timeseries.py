import argparse
import os
import os.path as op
from nilearn.input_data import NiftiMasker
from nilearn import image
import pandas as pd


def main(subject, session, bids_folder='/data'):

    target_dir = op.join(bids_folder, 'derivatives', 'extracted_signal', f'sub-{subject}', f'ses-{session}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)


    data = [op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', f'ses-{session}', 'func',
              f'sub-{subject}_ses-{session}_task-task_run-{run}_space-T1w_desc-preproc_bold.nii.gz')
        for run in range(1,9)]

    mask = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-wang15ips_mask.nii.gz')

    mask = image.resample_to_img(mask, data[0], interpolation='nearest')

    masker = NiftiMasker(mask)

    data = pd.concat([pd.DataFrame(masker.fit_transform(d)) for d in data],
                    keys=range(1, 9), names=['run'])
    data.index.set_names('frame', 1)
    print(data)

    data.to_csv(op.join(target_dir, f'sub-{subject}_ses-{session}_desc-ips_timeseries.tsv'), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder, )
