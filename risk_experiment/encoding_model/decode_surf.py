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
from risk_experiment.utils import get_single_trial_surf_data, get_surf_mask, get_prf_parameters
import numpy as np


stimulus_range = np.linspace(np.log(5), np.log(80), 100)
mask = 'wang15_ips'
space = 'fsnative'

def main(subject, session, smoothed, n_verts=100, bids_folder='/data',
        mask='wang15_ips'):

    target_dir = op.join(bids_folder, 'derivatives', 'decoded_pdfs')

    if smoothed:
        target_dir += '.smoothed'

    target_dir = op.join(target_dir, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)


    paradigm = [pd.read_csv(op.join(bids_folder, f'sub-{subject}', f'ses-{session}',
                               'func', f'sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'), sep='\t')
                for run in range(1, 9)]
    paradigm = pd.concat(paradigm, keys=range(1,9), names=['run']).droplevel(1)
    paradigm = paradigm[paradigm.trial_type == 'stimulus 1'].set_index('trial_nr', append=True)

    paradigm['log(n1)'] = np.log(paradigm['n1'])
    print(paradigm)

    data = get_single_trial_surf_data(subject, session, bids_folder, mask=mask, smoothed=smoothed,
            space=space)
    data.index = paradigm.index


    # np.random.seed(666)
    # resample_mask = np.random.choice(data.columns, n_verts)
    # data = data[resample_mask].astype(np.float32)

    pdfs = []
    runs = range(1, 9)

    for test_run in runs:

        test_data, test_paradigm = data.loc[test_run].copy(), paradigm.loc[test_run].copy()
        train_data, train_paradigm = data.drop(test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        pars = get_prf_parameters(subject, session, run=test_run, mask=mask, bids_folder=bids_folder, smoothed=smoothed,space=space)

        # pars = pars.loc[resample_mask]

        model = GaussianPRF(parameters=pars)
        pred = model.predict(paradigm=train_paradigm['log(n1)'].astype(np.float32))

        r2 = get_rsq(train_data, pred)
        print(r2.describe())
        print(r2.sort_values(ascending=False))
        r2_mask = r2.sort_values(ascending=False).index[:n_verts]
        model.apply_mask(r2_mask)

        train_data = train_data[r2_mask].astype(np.float32)
        test_data = test_data[r2_mask].astype(np.float32)

        print(model.parameters)
        print(train_data)

        model.init_pseudoWWT(stimulus_range, model.parameters)
        residfit = ResidualFitter(model, train_data,
                                  train_paradigm['log(n1)'].astype(np.float32))

        omega, dof = residfit.fit(init_sigma2=10.0,
                method='t',
                max_n_iterations=10000)

        print('DOF', dof)

        bins = np.linspace(np.log(5), np.log(80), 150, endpoint=True).astype(np.float32)

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

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_mask-{mask}_nverts-{n_verts}_space-{space}_pars.tsv')
    pdfs.to_csv(target_fn, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--mask', default='wang15_ips')
    parser.add_argument('--n_verts', default=100, type=int)
    args = parser.parse_args()

    main(args.subject, args.session, args.smoothed, args.n_verts,
            bids_folder=args.bids_folder, mask=args.mask)
            
# def main(subject, session, smoothed, n_verts=100, bids_folder='/data',
        # mask='wang15_ips'):