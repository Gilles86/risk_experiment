import argparse
import pandas as pd
from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter
from risk_experiment.utils import get_surf_data, get_mapper_paradigm, write_gifti, get_target_dir
from risk_experiment.utils.surface import transform_data
from nilearn import surface

import os
import os.path as op
import numpy as np
import seaborn as sns


def main(subject, session, bids_folder='/data/ds-risk'):

    target_dir = 'encoding_model'
    target_dir = get_target_dir(subject, session, bids_folder, target_dir)

    paradigm = [pd.read_csv(op.join(bids_folder, f'sub-{subject}', f'ses-{session}',
                               'func', f'sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'), sep='\t')
                for run in range(1, 9)]
    paradigm = pd.concat(paradigm, keys=range(1,9), names=['run'])
    paradigm = paradigm[paradigm.trial_type == 'stimulus 1'][['n1', 'trial_nr']].set_index('trial_nr')

    for hemi in ['L', 'R']:
        data = surface.load_surf_data(op.join(bids_folder, 'derivatives', 'glm_stim1_surf',
                                              f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-task_space-fsnative_desc-stims1_hemi-{hemi}.pe.gii')).T
        model = GaussianPRF()
        # SET UP GRID
        mus = np.log(np.linspace(5, 80, 100, dtype=np.float32))
        sds = np.log(np.linspace(2, 30, 100, dtype=np.float32))
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)

        optimizer = ParameterFitter(model, data, paradigm)

        grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines, use_correlation_cost=True)
        grid_parameters = optimizer.refine_baseline_and_amplitude(grid_parameters, n_iterations=5)


        optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
                r2_atol=0.00001)

        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.optim_space-fsnative_hemi-{hemi}.func.gii')
        write_gifti(subject, session, bids_folder, 'fsnative', 
                pd.concat([optimizer.r2], keys=[hemi], names=['hemi']), target_fn)
        transform_data(target_fn, f'sub-{subject}', bids_folder, target_subject='fsaverage')

        for par, values in optimizer.estimated_parameters.T.iterrows():
            print(values)
            target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim_space-fsnative_hemi-{hemi}.func.gii')
            write_gifti(subject, session, bids_folder, 'fsnative', 
                    pd.concat([values], keys=[hemi], names=['hemi']), target_fn)
            transform_data(target_fn, f'sub-{subject}', bids_folder, target_subject='fsaverage')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder)
