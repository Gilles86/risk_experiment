
import argparse
import pandas as pd
from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter
from risk_experiment.utils import get_surf_data, get_mapper_paradigm, write_gifti, get_target_dir
from risk_experiment.utils.surface import transform_data
from nilearn import surface
from nilearn.input_data import NiftiMasker
from nilearn import image

import os
import os.path as op
import numpy as np
import seaborn as sns


def main(subject, session, bids_folder='/data/ds-risk', smoothed=False,
        stimulus='1',
        allow_neg=False,
        pca_confounds=False):

    key = 'glm_stim1'
    target_dir = 'encoding_model'

    if allow_neg:
        target_dir += '.posneg'

    if smoothed:
        key += '.smoothed'
        target_dir += '.smoothed'

    if pca_confounds:
        target_dir += '.pca_confounds'
        key += '.pca_confounds'

    target_dir = get_target_dir(subject, session, bids_folder, target_dir)

    paradigm = [pd.read_csv(op.join(bids_folder, f'sub-{subject}', f'ses-{session}',
                               'func', f'sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'), sep='\t')
                for run in range(1, 9)]
    paradigm = pd.concat(paradigm, keys=range(1,9), names=['run'])
    paradigm['log(n1)'] = np.log(paradigm['n1'])
    paradigm['log(n2)'] = np.log(paradigm['n2'])

    if stimulus  == '1':
        paradigm = paradigm[paradigm.trial_type == 'stimulus 1'].set_index('trial_nr')
        paradigm = np.log(paradigm['n1'])
    if stimulus == '2': 
        paradigm = paradigm[paradigm.trial_type == 'stimulus 2'].set_index('trial_nr')
        paradigm = np.log(paradigm['n2'])
    elif stimulus == 'both':

        p1 = paradigm[paradigm.trial_type == 'stimulus 1'].set_index('trial_nr')
        p2 = paradigm[paradigm.trial_type == 'stimulus 2'].set_index('trial_nr')
        paradigm = pd.concat((np.log(p1['n1']), np.log(p2['n2'])), keys=['stim1', 'stim2'], names=['stimulus'])

    model = GaussianPRF()
    # SET UP GRID
    mus = np.log(np.linspace(5, 80, 60, dtype=np.float32))
    sds = np.log(np.linspace(2, 30, 60, dtype=np.float32))
    if allow_neg:
        amplitudes = np.array([-1., 1.], dtype=np.float32)
    else:
        amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    mask = op.join(bids_folder, 'derivatives', f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')

    masker = NiftiMasker(mask_img=mask)

    data_folder = op.join(bids_folder, 'derivatives', key,
                                              f'sub-{subject}', f'ses-{session}', 'func')
    if stimulus == '1':
        data = op.join(data_folder, f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims1_pe.nii.gz')
    elif stimulus == '2':
        data = op.join(data_folder, f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims2_pe.nii.gz')
    elif stimulus == 'both':
        d1 = op.join(data_folder, f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims1_pe.nii.gz')
        d2 = op.join(data_folder, f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims2_pe.nii.gz')
        data = image.concat_imgs([d1, d2])

    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)
    print(data)

    data = pd.DataFrame(data, index=paradigm.index)

    optimizer = ParameterFitter(model, data, paradigm)

    grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines, use_correlation_cost=True)
    print('grid', grid_parameters.describe())
    grid_parameters = optimizer.refine_baseline_and_amplitude(grid_parameters, n_iterations=2, positive_amplitude=(not allow_neg))
    print('reifned', grid_parameters.describe())


    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
            r2_atol=0.00001)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.optim_space-T1w_pars.nii.gz')

    masker.inverse_transform(optimizer.r2).to_filename(target_fn)

    for par, values in optimizer.estimated_parameters.T.iterrows():
        print(values)
        # target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_stim-{stimulus}_desc-{par}.optim_space-T1w_pars.nii.gz')
        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim_space-T1w_pars.nii.gz')
        masker.inverse_transform(values).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pca_confounds', action='store_true')
    parser.add_argument('--stimulus', default='1')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder, smoothed=args.smoothed,
            pca_confounds=args.pca_confounds,
            stimulus=args.stimulus)
