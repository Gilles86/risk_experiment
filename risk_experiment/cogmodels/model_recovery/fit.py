import argparse
import os.path as op
import pandas as pd
from bauer.models import RiskModel, ExpectedUtilityRiskModel


def load_simulated_data(model_label, simulation_index, bids_folder='/data/ds-risk'):
    fn = op.join(bids_folder, 'derivatives', 'cogmodels', 'model_recovery', f'simulated_data_{model_label}_{simulation_index}.tsv',)
    df = pd.read_csv(fn, sep='\t', index_col=[0, 1, 2])

    df['choice'] = df['simulated_choice']
    df['chose_risky'] = df['choice'].where(~df['risky_first'], ~df['choice'])

    return df

def main(generating_model, recovering_model, simulation_index, bids_folder='/data/ds-risk'):

    assert generating_model in ['eu', 'klw', 'pmrc'], 'Model not implemented'
    assert recovering_model in ['eu', 'klw', 'pmrc'], 'Model not implemented'

    target_dir = op.join(bids_folder, 'derivatives', 'cogmodels', 'model_recovery', 'traces')

    df = load_simulated_data(generating_model, simulation_index, bids_folder)

    model = build_model(df, recovering_model)

    target_accept = 0.9

    idata = model.sample(target_accept=target_accept, raws=1500, tune=1500, )

    idata.to_netcdf(op.join(target_dir, f'generating-{generating_model}_recovering-{recovering_model}_{simulation_index}.netcdf'))

def build_model(df, recovering_model):
    if recovering_model == 'eu':
        model = ExpectedUtilityRiskModel(df)
    elif recovering_model =='klw':
        model = RiskModel(df, fit_seperate_evidence_sd=False, prior_estimate='klw')
    elif recovering_model == 'pmrc':
        model = RiskModel(df, fit_seperate_evidence_sd=True, prior_estimate='full')

    model.build_estimation_model()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit a cognitive model to data')
    parser.add_argument('generating_model', type=str, help='The label of the model that generated the data', choices=['eu', 'klw', 'pmrc'])
    parser.add_argument('recovering_model', type=str, help='The label of the model that will recover the data', choices=['eu', 'klw', 'pmrc'])
    parser.add_argument('simulation_index', type=int, help='The index of the simulation', choices=range(1, 11))
    parser.add_argument('--bids_folder', default='/data/ds-risk', type=str, help='Path to BIDS folder')
    args = parser.parse_args()
    main(args.generating_model, args.recovering_model, args.simulation_index, args.bids_folder)