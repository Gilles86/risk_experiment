import argparse
import os
import pingouin
import numpy as np
import os.path as op
import pandas as pd
from nilearn import surface
from braincoder.optimize import ResidualFitter
from braincoder.models import GaussianPRF, LogGaussianPRF
from braincoder.utils import get_rsq
import numpy as np
from risk_experiment.utils import Subject


mask = 'wang15_ips'
space = 'T1w'

def main(subject, session, smoothed, pca_confounds, n_voxels=1000, bids_folder='/data',
denoise=False, retroicor=False, mask='wang15_ips', natural_space=False):

    target_dir = op.join(bids_folder, 'derivatives', 'decoded_pdfs.volume')

    if denoise:
        target_dir += '.denoise'

    if (retroicor) and (not denoise):
        raise Exception("When not using GLMSingle RETROICOR is *always* used!")

    if retroicor:
        target_dir += '.retroicor'

    if smoothed:
        target_dir += '.smoothed'

    if pca_confounds:
        target_dir += '.pca_confounds'

    if natural_space:
        target_dir += '.natural_space'

    target_dir = op.join(target_dir, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    sub = Subject(subject, bids_folder)
    paradigm = sub.get_behavior(sessions=session, drop_no_responses=False)
    paradigm['log(n1)'] = np.log(paradigm['n1'])
    paradigm = paradigm.droplevel(['subject', 'session'])

    if natural_space:
        paradigm = paradigm['n1']
        stimulus_range = np.arange(5, 28*4+1)
    else:
        paradigm = paradigm['log(n1)']
        stimulus_range = np.linspace(np.log(5), np.log(28*4+1), 200)

    data = sub.get_single_trial_volume(session, roi=mask, smoothed=smoothed, pca_confounds=pca_confounds, denoise=denoise, retroicor=retroicor).astype(np.float32)
    data.index = paradigm.index
    print(data)

    pdfs = []
    runs = range(1, 9)

    for test_run in runs:

        test_data, test_paradigm = data.loc[test_run].copy(), paradigm.loc[test_run].copy()
        train_data, train_paradigm = data.drop(test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        pars = sub.get_prf_parameters_volume(session, cross_validated=True,
        denoise=denoise, retroicor=retroicor,
                smoothed=smoothed, pca_confounds=pca_confounds,
                run=test_run, roi=mask, natural_space=natural_space)
        print(pars)


        if natural_space:
            model = LogGaussianPRF(parameters=pars)
        else:
            model = GaussianPRF(parameters=pars)

        pred = model.predict(paradigm=train_paradigm.astype(np.float32))

        r2 = get_rsq(train_data, pred)
        print(r2.describe())
        r2_mask = r2.sort_values(ascending=False).index[:n_voxels]

        train_data = train_data[r2_mask]
        test_data = test_data[r2_mask]

        print(r2.loc[r2_mask])
        model.apply_mask(r2_mask)

        model.init_pseudoWWT(stimulus_range, model.parameters)
        residfit = ResidualFitter(model, train_data,
                                  train_paradigm.astype(np.float32))

        omega, dof = residfit.fit(init_sigma2=10.0,
                method='t',
                max_n_iterations=10000)

        print('DOF', dof)

        bins = stimulus_range.astype(np.float32)

        pdf = model.get_stimulus_pdf(test_data, bins,
                model.parameters,
                omega=omega,
                dof=dof)

        pdf /= np.trapz(pdf, pdf.columns,axis=1)[:, np.newaxis]
        E = pd.Series(np.trapz(pdf*pdf.columns.values[np.newaxis,:], pdf.columns, axis=1), index=pdf.index)

        print(pd.concat((E, test_paradigm), axis=1))
        print(pingouin.corr(E, test_paradigm))

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
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--retroicor', action='store_true')
    parser.add_argument('--mask', default='npcr')
    parser.add_argument('--n_voxels', default=100, type=int)
    parser.add_argument('--natural_space', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, args.smoothed, args.pca_confounds,
            args.n_voxels,
            denoise=args.denoise,
            retroicor=args.retroicor,
            bids_folder=args.bids_folder, mask=args.mask,
            natural_space=args.natural_space)
