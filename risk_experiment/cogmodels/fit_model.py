import argparse
from scipy.stats import zscore
# from risk_experiment.cogmodels.evidence_model import (EvidenceModel,
#                                                EvidenceModelTwoPriors, EvidenceModelGauss, EvidenceModelDiminishingUtility,
#                                                EvidenceModelTwoPriorsDiminishingUtility,
#                                                EvidenceModelDifferentEvidence,
#                                                EvidenceModelDifferentEvidenceTwoPriors, WoodfordModel)

from bauer.models import RiskModel, RiskRegressionModel, RiskLapseRegressionModel
from risk_experiment.utils.data import get_all_behavior
import os
import os.path as op
import arviz as az


def main(model_label, session, burnin=1500, samples=1000, bids_folder='/data/ds-risk'):

    df = get_data(model_label, session, bids_folder)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.9

    if model_label == 'certainty_evidence':
        target_accept = 0.925

    model = build_model(model_label, df)

    trace = model.sample(burnin, samples, target_accept=target_accept)
    az.to_netcdf(trace,
                 op.join(target_folder, f'ses-{session}_model-{model_label}_trace.netcdf'))

def get_data(model_label, session, bids_folder):
    df = get_all_behavior(bids_folder=bids_folder)
    df = df.xs(session, 0, 'session')
    df['choice'] = df['choice'] == 2.0

    if model_label.startswith('certainty'):
        df = df[~df.z_uncertainty.isnull()]

    return df

def build_model(model_label, df):
    if model_label == '1':
        model = RiskModel(df, 'full')
    elif model_label == 'certainty_full':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'z_uncertainty', 'n2_evidence_sd':'z_uncertainty', 'risky_prior_mu':'z_uncertainty',
        'risky_prior_std':'z_uncertainty', 'safe_prior_mu':'z_uncertainty', 'safe_prior_std':'z_uncertainty'})
    elif model_label == 'certainty_evidence':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'z_uncertainty', 'n2_evidence_sd':'z_uncertainty',})
    elif model_label == 'certainty_lapse':
        model = RiskLapseRegressionModel(df, prior_estimate='full', regressors={'p_lapse':'z_uncertainty'})
    elif model_label == 'certainty_hybrid':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'z_uncertainty', 'n2_evidence_sd':'z_uncertainty', 'p_lapse':'z_uncertainty'})
    else:
        raise Exception(f'Do not know model label {model_label}')

    model.build_estimation_model()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(args.model_label, args.session, bids_folder=args.bids_folder)
