from bauer.models import RiskModel
import numpy as np
import pandas as pd
import argparse
import os.path as op
import arviz as az
from risk_experiment.utils import get_all_behavior

def main(simulation_ix, n_subjects, bids_folder, parameter_source='prior', use_calibrated_design=False):

    model = RiskModel(fit_seperate_evidence_sd=True, prior_estimate='full')

    if parameter_source == 'prior':
        parameters = model.sample_parameters_from_prior(n_subjects)
        paradigm = model.get_example_paradigm(n_subjects=n_subjects)
    elif parameter_source == 'data':
        pars = model.free_parameters.keys()
        idata_full = az.from_netcdf(op.join(bids_folder, 'derivatives', 'cogmodels', 'model-12_trace.netcdf'))
        pars_full_3t = pd.concat([model.forward_transform(idata_full.posterior[par].to_dataframe().xs('Intercept', 0, f'{par}_regressors'), par) for par in pars], axis=1).groupby('subject').mean()
        pars_full_7t = pd.concat([model.forward_transform(idata_full.posterior[par].to_dataframe().unstack(level=-1).sum(1).to_frame(par), par) for par in pars], axis=1).groupby('subject').mean()
        pars_full_3t.index = pars_full_3t.index.astype(int)
        pars_full_7t.index = pars_full_7t.index.astype(int)

        parameters = pd.concat((pars_full_3t, pars_full_7t), keys=['3t', '7t'], names=['session'], axis=0)
        parameters = parameters.groupby('subject').mean()

        # Make sure the level 'subject' is an integer
        parameters.index = parameters.index.astype(int)

        if use_calibrated_design:
            paradigm = get_all_behavior(bids_folder=bids_folder)
            paradigm = paradigm[['n1', 'n2', 'p1', 'p2']]
            paradigm.index = paradigm.index.set_levels(paradigm.index.levels[0].astype(int), level=0)
        else:
            paradigm = model.get_example_paradigm(n_subjects=32)
            paradigm = paradigm.loc[parameters.index.unique()]

    print(parameters)
    print(paradigm)

    if use_calibrated_design:
        print('Calibrated design')
        df = model.simulate(paradigm, parameters, n_samples=1)
    else:
        print('Uncalibrated design')
        df = model.simulate(paradigm, parameters, n_samples=4)

    df['choice'] = df['simulated_choice']
    print(df)

    key = f'source-{parameter_source}'
    if use_calibrated_design:
        key += '_calibrateddesign'
    key += f'_{simulation_ix}'
    
    target_dir = op.join(bids_folder, 'derivatives', 'parameter_recovery')
    parameters.to_csv(op.join(target_dir, f'simulated_parameters_{key}.tsv'), sep='\t')
    df.to_csv(op.join(target_dir, f'simulated_data_{key}.tsv'), sep='\t')


    model.build_estimation_model(df)
    idata = model.sample()
    idata.to_netcdf(op.join(target_dir, f'posterior_samples_{key}.nc'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate data for parameter recovery')
    parser.add_argument('simulation_ix', type=int, help='Index of simulation')
    parser.add_argument('n_subjects', type=int, help='Number of subjects to simulate')
    parser.add_argument('--bids_folder', type=str, help='Path to BIDS folder')
    parser.add_argument('--parameter_source', default='prior', type=str, help='Source of parameters (prior or posterior)')
    parser.add_argument('--use_calibrated_design', action='store_true', help='Use calibrated design for simulation')

    args = parser.parse_args()

    main(args.simulation_ix, args.n_subjects, args.bids_folder, args.parameter_source, args.use_calibrated_design)