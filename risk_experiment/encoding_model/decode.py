import argparse
import os
import pingouin
import numpy as np
import os.path as op
import pandas as pd
from nilearn import surface
from braincoder.optimize import ResidualFitter
from braincoder.models import GaussianPRF
from braincoder.utils import get_rsq
from risk_experiment.utils import get_single_trial_volume, get_surf_mask, get_prf_parameters_volume, Subject
import numpy as np


stimulus_range = np.linspace(0, 6, 1000)
# stimulus_range = np.log(np.arange(400))
mask = 'wang15_ips'
space = 'T1w'

def main(subject, session, smoothed, pca_confounds, n_voxels=1000, bids_folder='/data',
        mask='wang15_ips'):

    target_dir = op.join(bids_folder, 'derivatives', 'decoded_pdfs.volume')

    if smoothed:
        target_dir += '.smoothed'

    if pca_confounds:
        target_dir += '.pca_confounds'

    target_dir = op.join(target_dir, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    sub = Subject(subject, bids_folder)
    paradigm = sub.get_behavior(sessions=session, drop_no_responses=False)
    paradigm['log(n1)'] = np.log(paradigm['n1'])
    paradigm = paradigm.droplevel(['subject', 'session'])

    data = get_single_trial_volume(subject, session, bids_folder=bids_folder, mask=mask, smoothed=smoothed, pca_confounds=pca_confounds).astype(np.float32)
    data.index = paradigm.index
    print(data)

    pdfs = []
    runs = range(1, 9)

    for test_run in runs:

        test_data, test_paradigm = data.loc[test_run].copy(), paradigm.loc[test_run].copy()
        train_data, train_paradigm = data.drop(test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        pars = get_prf_parameters_volume(subject, session, cross_validated=True,
                smoothed=smoothed, pca_confounds=pca_confounds,
                run=test_run, mask=mask, bids_folder=bids_folder)
        # pars = get_prf_parameters_volume(subject, session, cross_validated=False,  mask=mask, bids_folder=bids_folder)
        print(pars)

        model = GaussianPRF(parameters=pars)
        pred = model.predict(paradigm=train_paradigm['log(n1)'].astype(np.float32))

        r2 = get_rsq(train_data, pred)
        print(r2.describe())
        r2_mask = r2.sort_values(ascending=False).index[:n_voxels]

        train_data = train_data[r2_mask]
        test_data = test_data[r2_mask]

        print(r2.loc[r2_mask])
        model.apply_mask(r2_mask)

        model.init_pseudoWWT(stimulus_range, model.parameters)
        residfit = ResidualFitter(model, train_data,
                                  train_paradigm['log(n1)'].astype(np.float32))

        omega, dof = residfit.fit(init_sigma2=10.0,
                method='t',
                max_n_iterations=10000)

        print('DOF', dof)

        bins = stimulus_range.astype(np.float32)

        pdf = model.get_stimulus_pdf(test_data, bins,
                model.parameters,
                omega=omega,
                dof=dof)


        print(pdf)
        E = (pdf * pdf.columns).sum(1) / pdf.sum(1)

        print(pd.concat((E, test_paradigm['log(n1)']), axis=1))
        print(pingouin.corr(E, test_paradigm['log(n1)']))

        pdfs.append(pdf)

    pdfs = pd.concat(pdfs)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_mask-{mask}_nvoxels-{n_voxels}_space-{space}_pars.tsv')
    pdfs.to_csv(target_fn, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pca_confounds', action='store_true')
    parser.add_argument('--mask', default='wang15_ips')
    parser.add_argument('--n_voxels', default=100, type=int)
    args = parser.parse_args()

    main(args.subject, args.session, args.smoothed, args.pca_confounds,
            args.n_voxels,
            bids_folder=args.bids_folder, mask=args.mask)
