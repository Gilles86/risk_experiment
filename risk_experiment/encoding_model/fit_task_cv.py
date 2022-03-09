import argparse
import pandas as pd
from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq
from risk_experiment.utils import get_surf_data, get_mapper_paradigm, write_gifti, get_target_dir
from risk_experiment.utils.surface import transform_data
from nilearn import surface
from nilearn.input_data import NiftiMasker

import os
import os.path as op
import numpy as np
import seaborn as sns


def main(subject, session, bids_folder='/data/ds-risk', smoothed=False)
         

    key = 'glm_stim1'
    target_dir = 'encoding_model'

    if smoothed:
        key += '.smoothed'
        target_dir += '.smoothed'

    target_dir = get_target_dir(subject, session, bids_folder, target_dir)

    paradigm = [pd.read_csv(op.join(bids_folder, f'sub-{subject}', f'ses-{session}',
                                    'func', f'sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'), sep='\t')
                for run in range(1, 9)]
    paradigm = pd.concat(paradigm, keys=range(1, 9), names=['run'])
    paradigm = paradigm[paradigm.trial_type ==
                        'stimulus 1'].set_index('trial_nr')

    paradigm['log(n1)'] = np.log(paradigm['n1'])
    paradigm = paradigm['log(n1)']

    model = GaussianPRF()
    # SET UP GRID
    mus = np.log(np.linspace(5, 80, 100, dtype=np.float32))
    sds = np.log(np.linspace(2, 30, 100, dtype=np.float32))
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    mask = op.join(bids_folder, 'derivatives',
                   f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')

    masker = NiftiMasker(mask_img=mask)

    data = op.join(bids_folder, 'derivatives', key,
                   f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims1_pe.nii.gz')

    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)
    print(data)

    data = pd.DataFrame(data, index=paradigm.index)

    for test_run in range(1, 9):

        test_data, test_paradigm = data.loc[test_run].copy(
        ), paradigm.loc[test_run].copy()
        print(test_data, test_paradigm)
        train_data, train_paradigm = data.drop(
            test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        optimizer = ParameterFitter(model, train_data, train_paradigm)

        grid_parameters = optimizer.fit_grid(
            mus, sds, amplitudes, baselines, use_correlation_cost=True)
        grid_parameters = optimizer.refine_baseline_and_amplitude(
            grid_parameters, n_iterations=2)

        optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
                      r2_atol=0.00001)

        target_fn = op.join(
            target_dir, f'sub-{subject}_ses-{session}_run-{test_run}_desc-r2.optim_space-T1w_pars.nii.gz')

        masker.inverse_transform(optimizer.r2).to_filename(target_fn)

        cv_r2 = get_rsq(test_data, model.predict(parameters=optimizer.estimated_parameters,
                                                 paradigm=test_paradigm.astype(np.float32)))

        target_fn = op.join(
            target_dir, f'sub-{subject}_ses-{session}_run-{test_run}_desc-cvr2.optim_space-T1w_pars.nii.gz')

        masker.inverse_transform(cv_r2).to_filename(target_fn)

        for par, values in optimizer.estimated_parameters.T.iterrows():
            print(values)
            target_fn = op.join(
                target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim_space-T1w_pars.nii.gz')

            masker.inverse_transform(values).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder, smoothed=args.smoothed)
