import argparse
import os
import pingouin
import numpy as np
import os.path as op
import pandas as pd
from nilearn import surface
from braincoder.optimize import ResidualFitter, ParameterFitter
from braincoder.models import GaussianPRF
from braincoder.utils import get_rsq
from risk_experiment.utils import get_single_trial_volume, get_surf_mask, get_prf_parameters_volume, get_task_behavior, get_all_task_behavior
import numpy as np


subject = '04'
session1 = '3t1'
session2 = '3t2'
mask = 'npcr'
bids_folder = '/data'
n_voxels = 250

stimulus_range = np.linspace(np.log(5), np.log(80), 100)

def main(subject, session, n_voxels=250, bids_folder='/data', mask='wang15_ips'):
    
    session1 = session[:2] + '1'
    session2 = session[:2] + '2'


    pars = get_prf_parameters_volume(subject, session1, 
            cross_validated=False,
            mask=mask,bids_folder=bids_folder).astype(np.float32)

    behavior = get_task_behavior(subject, session2, bids_folder)
    data = get_single_trial_volume(subject, session2, bids_folder=bids_folder, mask=mask).astype(np.float32)
    print(data)

    paradigm = behavior[['log(n1)']].astype(np.float32)
    paradigm.index = data.index
    print(paradigm)


    pdfs = []
    runs = range(1, 9)

    for test_run in runs:

        test_data, test_paradigm = data.xs(test_run, level='run').copy(), paradigm.xs(test_run, level='run').copy()
        train_data, train_paradigm = data.drop(test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        model = GaussianPRF(parameters=pars, paradigm=train_paradigm)
        parfitter = ParameterFitter(model, train_data, train_paradigm)

        new_pars = parfitter.refine_baseline_and_amplitude(pars)
        new_pars = parfitter.fit(init_pars=new_pars, fixed_pars=['mu', 'sd'])
        print(new_pars)
        model.parameters = new_pars.astype(np.float32)

        pred = model.predict()
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

    target_dir = op.join(bids_folder, 'derivatives', 'decoded_pdfs.volume.across_session')
    target_dir = op.join(target_dir, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session2}_mask-{mask}_nvoxels-{n_voxels}_space-{space}_pars.tsv')
    pdfs.to_csv(target_fn, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--mask', default='wang15_ips')
    parser.add_argument('--n_voxels', default=100, type=int)
    args = parser.parse_args()

    main(args.subject, args.session, n_voxels=args.n_voxels,
            bids_folder=args.bids_folder, mask=args.mask)
            
