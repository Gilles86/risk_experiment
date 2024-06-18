
import argparse
import pandas as pd
from risk_experiment.cogmodels.fit_model import build_model, get_data
from bauer.models import ExpectedUtilityRiskModel, RiskModel
import arviz as az
import os
import os.path as op


def main(model_label, n_simulations=10, bids_folder='/data/ds-risk'):

    assert model_label in ['eu', 'klw', 'pmrc'], 'Model not implemented'

    if model_label == 'pmrc':
        print('yo')
        model_label = '12'

    target_dir = op.join(bids_folder, 'derivatives', 'cogmodels', 'model_recovery')
    os.makedirs(target_dir, exist_ok=True)

    # Get the model and data
    data = get_data(model_label, None, bids_folder, None)
    model = build_model(model_label, data, None)
    idata = az.from_netcdf(f'/data/ds-risk/derivatives/cogmodels/model-{model_label}_trace.netcdf')

    pars = model.free_parameters.keys()
    pars_full_3t = pd.concat([model.forward_transform(idata.posterior[par].to_dataframe().xs('Intercept', 0, f'{par}_regressors'), par) for par in pars], axis=1).groupby('subject').mean()
    pars_full_7t = pd.concat([model.forward_transform(idata.posterior[par].to_dataframe().unstack(level=-1).sum(1).to_frame(par), par) for par in pars], axis=1).groupby('subject').mean()
    pars_full_3t.index = pars_full_3t.index.astype(int)
    pars_full_7t.index = pars_full_7t.index.astype(int)

    parameters = pd.concat((pars_full_3t, pars_full_7t), keys=['3t', '7t'], names=['session'], axis=0)
    parameters = parameters.groupby('subject').mean()

    # Make sure the level 'subject' is an integer
    data.index = data.index.set_levels(data.index.levels[0].astype(int), level=0)

    if model_label == 'eu':
        simulation_model = ExpectedUtilityRiskModel(data)
    elif model_label == '12':
        simulation_model = RiskModel(data, fit_seperate_evidence_sd=True, prior_estimate='full')
    elif model_label == 'klw':
        simulation_model = RiskModel(data, fit_seperate_evidence_sd=False, prior_estimate='klw')

    simulated_data = simulation_model.simulate(data.droplevel([1, 2],axis=0), parameters, n_samples=n_simulations)

    model_label = 'pmrc' if model_label == '12' else 'klw'

    for sample, d in simulated_data.groupby('sample'):
        d.to_csv(op.join(target_dir, f'simulated_data_{model_label}_{sample}.tsv'), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit a cognitive model to data')
    parser.add_argument('model_label', type=str, help='The label of the model to fit')
    parser.add_argument('--n_simulations', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--bids_folder', default='/data/ds-risk', type=str, help='Path to BIDS folder')
    args = parser.parse_args()

    main(args.model_label, args.n_simulations, args.bids_folder)