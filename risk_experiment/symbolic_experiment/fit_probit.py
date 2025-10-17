from risk_experiment.symbolic_experiment.utils import get_subjects, get_behavioral_data
import argparse
from pathlib import Path
from bambi import Model
import pandas as pd
import numpy as np


def main(model_label, bids_folder, burnin=2000, samples=1000):

    target_dir = Path(bids_folder) / 'derivatives' / 'risk_model' / 'psychophysical'
    target_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_label, bids_folder)

    target_accept = .9
    n_cores = 4
    idata = model.fit(burnin, samples, init='adapt_diag', target_accept=target_accept, cores=n_cores)
    # Save the samples
    idata.to_netcdf(target_dir / f'model{model_label}_samples.nc')


def build_model(model_label=1, bids_folder='/data/ds-symbolicrisk'):

    df = get_data(bids_folder, model_label=model_label)

    if model_label == 0:
        model = Model('chose_risky ~ log_risky_safe + (log_risky_safe|subject)', df.reset_index('subject'), family='bernoulli', link='probit')
    elif model_label == 1:
        model = Model('chose_risky ~ log_risky_safe*order + (log_risky_safe*order|subject)', df.reset_index('subject'), family='bernoulli', link='probit')
    elif model_label in [2, 23]:
        model = Model('chose_risky ~ log_risky_safe*C(n_safe_bin)*order + (log_risky_safe*C(n_safe_bin)*order|subject)', df.reset_index('subject'), family='bernoulli', link='probit')
    elif model_label == 3:
        model = Model('chose_risky ~ log_risky_safe*C(n_safe_bin) + (log_risky_safe*C(n_safe_bin)|subject)', df.reset_index('subject'), family='bernoulli', link='probit')
    elif model_label == 4:
        model = Model('chose_risky ~ log_risky_safe*n_safe*order + (log_risky_safe*n_safe*order|subject)', df.reset_index('subject'), family='bernoulli', link='probit')
    else:
        raise ValueError(f'Unknown model label {model_label}')

    return model


def get_data(bids_folder='/data/ds-symbolicrisk', model_label=1):

    df= get_behavioral_data(bids_folder)
    df['p1'], df['p2'] = df['prob1'], df['prob2']

    df['log_risky_safe'] = df['log(risky/safe)']
    df['order'] = df['risky_first'].map({True:'Risky first', False: 'Safe first'})

    if model_label in [23]:
        n_risk_bins = 3
    else:
        n_risk_bins = 5

    bins = np.exp(np.linspace(np.log(5), np.log(28), n_risk_bins + 1))

    lower_bins = bins[:-1]
    lower_bins[0] = 5
    higher_bins = bins[1:]
    higher_bins[-1] = 28

    bin_labels = []
    for i in range(len(lower_bins)):
        bin_labels.append(f'{int(lower_bins[i])}-{int(higher_bins[i])}')

    df['n_safe_bin'] = pd.cut(df['n_safe'], bins=bins, include_lowest=True, labels=bin_labels)

    print(df['n_safe_bin'])

    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_label', type=int, default='/data/ds-symbolicrisk')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-symbolicrisk')

    args = parser.parse_args()

    main(model_label=args.model_label, bids_folder=args.bids_folder)

