import re
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
import pandas as pd
from risk_experiment.utils import get_all_subjects
from scipy.stats import zscore


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

    if model_label.startswith('neural'):
        decoding_info = pd.concat([sub.get_decoding_info(session, mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
        df = df.join(decoding_info)
        df['median_split(sd)'] = df.groupby(['subject', 'session'], group_keys=False)['sd'].apply(lambda d: d>d.quantile()).map({True:'High neural uncertainty', False:'Low neural uncertainty'})
        df['median_split(E)'] = df.groupby(['subject', 'session', 'n1'], group_keys=False)['E'].apply(lambda d: d>d.quantile()).map({True:'Higher decoded', False:'Lower decoded'})

    if model_label.startswith('pupil_baseline'):
        pupil_baseline = pd.read_csv(op.join(bids_folder, 'derivatives', 'pupil', 'model-n1_n2_n', 'pre_stim_baseline.tsv'), sep='\t')
        pupil_baseline['subject'] = pupil_baseline['subject'].map(lambda d: f'{d:02d}')
        pupil_baseline = pupil_baseline.set_index(['subject', 'run', 'trial_nr'])
        print(df)
        print(pupil_baseline)
        df = df.join(pupil_baseline)
        df['pupil'] = df.groupby(['subject'], group_keys=False)['pupil'].apply(zscore)
        df['median_split_pupil_baseline'] = df.groupby(['subject'], group_keys=False)['pupil'].apply(lambda d: d>d.quantile()).map({True:'High pre-baseline pupil dilation', False:'Low pre-baseline dilation'})

    if model_label.startswith('subcortical_'):
        reg = re.compile('subcortical_(?P<roi>.+)')
        roi = reg.match(model_label).group(1)

        roi_baseline = pd.read_csv(op.join(bids_folder, 'derivatives', 'roi_analysis', 'model-n1_n2_n', roi, f'ses-{session}_pre_stim_baseline.tsv'), sep='\t')
        roi_baseline['subject'] = roi_baseline['subject'].map(lambda d: f'{d:02d}')
        roi_baseline = roi_baseline.set_index(['subject', 'run', 'trial_nr'])
        print(df)
        print(roi_baseline)
        df = df.join(roi_baseline)
        df['median_split_subcortical_baseline'] = df.groupby(['subject'], group_keys=False)[roi].apply(lambda d: d>d.quantile()).map({True:'High pre-baseline subcortical activation', False:'Low pre-baseline subcortical activation'})

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
    elif model_label == 'neural1':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'sd', 'n2_evidence_sd':'sd'})
    elif model_label == 'neural2':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'sd'})
    elif model_label == 'neural3':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'sd', 'n2_evidence_sd':'sd', 'risky_prior_mu':'sd',
        'risky_prior_std':'sd', 'safe_prior_mu':'sd', 'safe_prior_std':'sd'}) 
    elif model_label == 'neural4':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_mu':'0+E', 'n1_evidence_sd':'sd', 'risky_prior_std':'sd', 'safe_prior_std':'sd'}) 
    elif model_label == 'pupil_baseline1':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'pupil', 'n2_evidence_sd':'pupil', 'risky_prior_mu':'pupil',
        'risky_prior_std':'pupil', 'safe_prior_mu':'pupil', 'safe_prior_std':'pupil'}) 
    elif model_label == 'pupil_baseline2':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_mu':'0+pupil', 'n2_evidence_mu':'0+pupil'}) 
    elif model_label.startswith('subcortical_'):
        reg = re.compile('subcortical_(?P<roi>.+)')
        roi = reg.match(model_label).group(1)
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':roi, 'n2_evidence_sd':roi, 'risky_prior_mu':roi,
        'risky_prior_std':roi, 'safe_prior_mu':roi, 'safe_prior_std':roi}) 
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
