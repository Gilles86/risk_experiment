from risk_experiment.symbolic_experiment.utils import get_subjects, get_behavioral_data
import argparse
from bauer.models import RiskModel, FlexibleNoiseRiskModel, ExpectedUtilityRiskModel
from pathlib import Path
import pandas as pd
import numpy as np
from .utils import get_n_safe_bins

def get_data(model_label=1, bids_folder='/data/ds-symbolicrisk'):

    df= get_behavioral_data(bids_folder)
    df['p1'], df['p2'] = df['prob1'], df['prob2']

    df['choice'] = df['choice'] == 2.0

    df['n_safe_bin'] = get_n_safe_bins(df['n_safe'], n_risk_bins=5)
    df['risky_safe_bin'] = pd.qcut(df['log(risky/safe)'], 6).apply(lambda d: d.mid).astype(np.float32)

    return df


def build_model(model_label=1, bids_folder='/data/ds-symbolicrisk'):

    df = get_data(model_label, bids_folder)

    if model_label == -1:
        model = RiskModel(df[~df.choice.isnull()], prior_estimate='full', fit_seperate_evidence_sd=False)
    elif model_label == 0:
        model = RiskModel(df[~df.choice.isnull()], prior_estimate='shared', fit_seperate_evidence_sd=False)
    elif model_label == 1:
        model = RiskModel(df[~df.choice.isnull()], prior_estimate='shared')
    elif model_label == 2:
        model = RiskModel(df[~df.choice.isnull()], prior_estimate='full')
    elif model_label == 3:
        model = FlexibleNoiseRiskModel(df[~df.choice.isnull()])
    elif model_label == 4:
        model = FlexibleNoiseRiskModel(df[~df.choice.isnull()], prior_estimate='shared')
    elif model_label == 5:
        model = ExpectedUtilityRiskModel(df, probability_distortion=False, save_trialwise_eu=False)

    model.build_estimation_model()

    return model



def main(model_label=1, bids_folder='/data/ds-symbolicrisk'):

    subjects = get_subjects(bids_folder)

    target_dir = Path(bids_folder) / 'derivatives' / 'risk_model' / 'nlc'
    target_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_label, bids_folder)

    target_accept = 0.85
    if model_label == 5:
        target_accept = 0.9

    idata = model.sample(tune=3000, draws=3000, target_accept=target_accept)

    # Save the samples
    idata.to_netcdf(target_dir / f'model{model_label}_samples.nc')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_label', type=int, default=1)
    parser.add_argument('--bids_folder', type=str, default='/data/ds-symbolicrisk')

    args = parser.parse_args()

    main(model_label=args.model_label, bids_folder=args.bids_folder)

