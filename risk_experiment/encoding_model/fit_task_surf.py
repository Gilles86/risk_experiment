import argparse
import pandas as pd
from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter

from nilearn import surface

import os
import os.path as op
import numpy as np
import seaborn as sns


def main(subject, session, hemi, bids_folder='/data/ds-risk'):

    data = surface.load_surf_data(op.join(bids_folder, 'derivatives', 'glm_stim1_surf',
                                          f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-task_space-fsnative_desc-stims1_hemi-{hemi}.pe.gii'))

    data = data.T

    paradigm = [pd.read_csv(op.join(bids_folder, f'sub-{subject}', f'ses-{session}',
                               'func', f'sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'), sep='\t')
                for run in range(1, 9)]
    paradigm = pd.concat(paradigm, keys=range(1,9), names=['run'])
    paradigm = paradigm[paradigm.trial_type == 'stimulus 1'][['n1', 'trial_nr']].set_index('trial_nr')

    model = GaussianPRF()

    # SET UP GRID
    mus = np.log(np.linspace(5, 80, 30, dtype=np.float32))
    sds = np.log(np.linspace(2, 30, 30, dtype=np.float32))
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)
    # mus = np.linspace(0, np.log(80), 20, dtype=np.float32)
    # sds = np.linspace(.01, 2, 15, dtype=np.float32)
    # amplitudes = np.linspace(1e-6, 10, 10, dtype=np.float32)
    # baselines = np.linspace(-2., 0., 4, endpoint=True, dtype=np.float32)

    optimizer = ParameterFitter(model, data, paradigm)

    grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines, use_correlation_cost=True)
    grid_parameters = optimizer.refine_baseline_and_amplitude(grid_parameters, n_iterations=5)


    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=1000,
            r2_atol=0.00001)
    print(optimizer.r2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('hemi', default=None)
    parser.add_argument('--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, args.hemi,bids_folder=args.bids_folder)
