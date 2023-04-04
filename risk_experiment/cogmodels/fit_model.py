import re
import argparse
from scipy.stats import zscore
# from risk_experiment.cogmodels.evidence_model import (EvidenceModel,
#                                                EvidenceModelTwoPriors, EvidenceModelGauss, EvidenceModelDiminishingUtility,
#                                                EvidenceModelTwoPriorsDiminishingUtility,
#                                                EvidenceModelDifferentEvidence,
#                                                EvidenceModelDifferentEvidenceTwoPriors, WoodfordModel)

from bauer.models import RiskModel, RiskRegressionModel, RiskLapseRegressionModel, RNPRegressionModel
from risk_experiment.utils.data import get_all_behavior, Subject
import os
import os.path as op
import arviz as az
import pandas as pd
from risk_experiment.utils import get_all_subjects, get_all_subject_ids
from scipy.stats import zscore


def main(model_label, session, burnin=1500, samples=1500, bids_folder='/data/ds-risk', roi=None):

    df = get_data(model_label, session, bids_folder, roi)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.9

    if model_label == 'certainty_evidence':
        target_accept = 0.925

    if model_label == '2':
        target_accept = 0.925

    if model_label == '222':
        target_accept = 0.925

    if model_label.startswith('neural3'):
        target_accept = 0.925

    if model_label.startswith('neural4'):
        target_accept = 0.925


    model = build_model(model_label, df, roi)

    trace = model.sample(draws=samples, tune=burnin, target_accept=target_accept)

    roi_str = f'_{roi}' if roi is not None else ''    

    if session is None:
        az.to_netcdf(trace,
                        op.join(target_folder, f'model-{model_label}{roi_str}_trace.netcdf'))
    else:
        az.to_netcdf(trace,
                        op.join(target_folder, f'ses-{session}_model-{model_label}{roi_str}_trace.netcdf'))

def get_data(model_label, session, bids_folder, roi):
    df = get_all_behavior(sessions=session, bids_folder=bids_folder)

    df['choice'] = df['choice'] == 2.0

    if model_label.startswith('certainty'):
        df = df[~df.z_uncertainty.isnull()]

    if model_label.startswith('neural'):
        if session is None:
            decoding_info_3t = pd.concat([sub.get_decoding_info('3t2', mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
            decoding_info_7t = pd.concat([sub.get_decoding_info('7t2', mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
            decoding_info = pd.concat((decoding_info_3t, decoding_info_7t))
        else:
            decoding_info = pd.concat([sub.get_decoding_info(session, mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])

        df = df.join(decoding_info)
        decoding_info['sd'] = decoding_info.groupby(['subject', 'session'], group_keys=False)['sd'].apply(zscore)
        df['median_split(sd)'] = df.groupby(['subject', 'session'], group_keys=False)['sd'].apply(lambda d: d>d.quantile()).map({True:'High neural uncertainty', False:'Low neural uncertainty'})
        df['median_split(E)'] = df.groupby(['subject', 'session', 'n1'], group_keys=False)['E'].apply(lambda d: d>d.quantile()).map({True:'Higher decoded', False:'Lower decoded'})
        df['median_split_sd'] = df['median_split(sd)']

    elif model_label.startswith('rnp_neural'):
        decoding_info = pd.concat([sub.get_decoding_info(session, mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
        decoding_info['sd'] = decoding_info.groupby('subject')['sd'].apply(zscore)

        df = df.join(decoding_info)
        df['median_split(sd)'] = df.groupby(['subject', 'session', 'n1', 'n2'], group_keys=False)['sd'].apply(lambda d: d>d.quantile()).map({True:'High neural uncertainty', False:'Low neural uncertainty'})
        df['median_split(E)'] = df.groupby(['subject', 'session', 'n1', 'n2'], group_keys=False)['E'].apply(lambda d: d>d.quantile()).map({True:'Higher decoded', False:'Lower decoded'})
        df['median_split_sd'] = df['median_split(sd)']

    elif model_label.startswith('pupil'):
        reg = re.compile('pupil.*_(?P<variable>n1_pre|n1_post|n1_postpre|n2_choice)$')
        variable = reg.match(model_label).group(1)
        pupil = pd.read_csv(op.join(bids_folder, 'derivatives', 'pupil', 'model-n1_n2_n', 'pupil_pre_post12.tsv'), sep='\t', index_col=[0,1,2,3], dtype={'subject':str})
        pupil['q(pupil)'] = pupil.groupby(['subject', 'event type', 'prepost'], group_keys=False)['pupil'].apply(lambda x: pd.qcut(x, 5, labels=['q1', 'q2', 'q3', 'q4', 'q5']))
        if variable == 'n1_pre':
            pupil = pupil.xs('n1', 0, 'event type').xs('pre', 0, 'prepost')
        elif variable == 'n1_post':
            pupil = pupil.xs('n1', 0, 'event type').xs('post', 0, 'prepost')
        elif variable == 'n1_postpre':
            pupil = pupil.xs('n1', 0, 'event type').xs('post-pre', 0, 'prepost')
        elif variable == 'n2_choice':
            pupil = pupil.xs('n2', 0, 'event type')
        df = df.join(pupil)
        df['median_split_pupil'] = df.groupby(['subject', 'n1', 'n2'], group_keys=False)['pupil'].apply(lambda d: d>d.quantile()).map({True:'High pupil dilation', False:'Low pupil dilation'})

    if model_label.startswith('subcortical_prestim'):
        if roi is None:
            raise Exception('Need to define ROI!')
        roi_baseline = pd.read_csv(op.join(bids_folder, 'derivatives', 'roi_analysis', 'model-n1_n2_n', roi, f'ses-{session}_pre_stim_baseline.tsv'), sep='\t')
        roi_baseline['subject'] = roi_baseline['subject'].map(lambda d: f'{d:02d}')
        roi_baseline = roi_baseline.set_index(['subject', 'run', 'trial_nr'])
        print(df)
        print(roi_baseline)
        df = df.join(roi_baseline)
        df['median_split_subcortical_baseline'] = df.groupby(['subject'], group_keys=False)[roi].apply(lambda d: d>d.quantile()).map({True:'High subcortical activation', False:'Low subcortical activation'})

    if model_label.startswith('subcortical_response'):
        if roi is None:
            raise Exception('Need to define ROI!')
        subjects = get_all_subjects(bids_folder)
        roi_responses = pd.concat([sub.get_roi_timeseries(session, roi, single_trial=True) for sub in subjects])
        df = df.join(roi_responses)
        df[roi] = df[roi].groupby(['subject', 'session']).apply(zscore)
        print(df)
        df['median_split_subcortical_baseline'] = df.groupby(['subject'], group_keys=False)[roi].apply(lambda d: d>d.quantile()).map({True:'High pre-baseline subcortical activation', False:'Low pre-baseline subcortical activation'})

    if model_label.startswith('neural32') or model_label.startswith('neural33') or model_label.startswith('12') or model_label.startswith('42') or model_label.startswith('222') or model_label.startswith('neural55'):
        df['session'] = df.index.get_level_values('session')


    return df

def build_model(model_label, df, roi):
    if model_label == '1':
        model = RiskModel(df, 'full')
    elif model_label == '2':
        model = RiskRegressionModel(df, prior_estimate='shared', regressors={'n1_evidence_sd':'risky_first', 'n2_evidence_sd':'risky_first'})
    elif model_label == '22':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'risky_first', 'n2_evidence_sd':'risky_first'})
    elif model_label == '222':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'risky_first*session', 'n2_evidence_sd':'risky_first*session',
        'risky_prior_std':'session', 'safe_prior_mu':'session', 'safe_prior_std':'session'})
    elif model_label == '3':
        model = RiskModel(df, 'full', incorporate_probability='before_inference')
    elif model_label == '4':
        model = RiskModel(df, prior_estimate='shared')
    elif model_label == '42':
        model = RiskRegressionModel(df, prior_estimate='shared', regressors={'n1_evidence_sd':'session', 'n2_evidence_sd':'session', 'prior_mu':'session',
        'prior_std':'session',})
    elif model_label == '12':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'session', 'n2_evidence_sd':'session', 'risky_prior_mu':'session',
        'risky_prior_std':'session', 'safe_prior_mu':'session', 'safe_prior_std':'session'})
    elif model_label == 'rnp1':
        model = RNPRegressionModel(df)
    elif model_label == 'rnp2':
        model = RNPRegressionModel(df, regressors={'gamma':'C(n_safe)*risky_first', 'rnp':'C(n_safe)*risky_first'})
    elif model_label == 'rnp3':
        model = RNPRegressionModel(df, regressors={'gamma':'0+risky_first', 'rnp':'0+risky_first'})
    elif model_label == 'rnp_neural1':
        model = RNPRegressionModel(df, regressors={'gamma':'C(n_safe)*risky_first*median_split_sd', 'rnp':'C(n_safe)*risky_first*median_split_sd'})
    elif model_label == 'rnp_neural2':
        model = RNPRegressionModel(df, regressors={'gamma':'median_split_sd', 'rnp':'median_split_sd'})
    elif model_label == 'rnp_neural3':
        model = RNPRegressionModel(df, regressors={'gamma':'risky_first*median_split_sd', 'rnp':'risky_first*median_split_sd'})
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
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'sd', 'n2_evidence_sd':'sd', 'risky_prior_mu':'sd',
        'risky_prior_std':'sd+sd:risky_first', 'safe_prior_mu':'sd', 'safe_prior_std':'sd+sd:risky_first'}) 
    elif model_label == 'neural32':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'sd*session', 'n2_evidence_sd':'sd*session', 'risky_prior_mu':'sd*session',
        'risky_prior_std':'sd*session', 'safe_prior_mu':'sd*session', 'safe_prior_std':'sd*session'}) 
    elif model_label == 'neural33':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'sd+session', 'n2_evidence_sd':'sd+session', 'risky_prior_mu':'sd+session',
        'risky_prior_std':'sd+session', 'safe_prior_mu':'sd+session', 'safe_prior_std':'sd+session'}) 
    elif model_label == 'neural33_null':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'session', 'n2_evidence_sd':'session', 'risky_prior_mu':'session',
        'risky_prior_std':'session', 'safe_prior_mu':'session', 'safe_prior_std':'session'}) 
    elif model_label == 'neural43':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'sd+session', 'n2_evidence_sd':'sd+session', 'risky_prior_mu':'sd+session',
        'risky_prior_std':'sd+session', 'safe_prior_mu':'sd+sd:risky_first+session', 'safe_prior_std':'sd+sd:risky_first+session'}) 
    elif model_label.startswith('subcortical_response1'):
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':roi, 'n2_evidence_sd':roi, 'risky_prior_mu':roi,
        'risky_prior_std':roi, 'safe_prior_mu':roi, 'safe_prior_std':roi}) 
    elif model_label == 'neural55':
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'sd+risky_first*session', 'n2_evidence_sd':'sd+risky_first*session',
        'risky_prior_std':'sd+session', 'safe_prior_mu':'sd+session', 'safe_prior_std':'sd+session'})
    elif model_label.startswith('subcortical_response2'):
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':roi, 'n2_evidence_sd':roi})
    elif model_label.startswith('subcortical_prestim'):
        model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':roi, 'n2_evidence_sd':roi, 'risky_prior_mu':roi,
        'risky_prior_std':roi, 'safe_prior_mu':roi, 'safe_prior_std':roi}) 
    elif model_label.startswith('pupil'):
        if model_label.startswith('pupil2'):
            model = RiskRegressionModel(df, prior_estimate='full', regressors={'risky_prior_std':'pupil', 'safe_prior_std':'pupil'})
        else:
            model = RiskRegressionModel(df, prior_estimate='full', regressors={'n1_evidence_sd':'pupil', 'n2_evidence_sd':'pupil', 'risky_prior_mu':'pupil',
            'risky_prior_std':'pupil', 'safe_prior_mu':'pupil', 'safe_prior_std':'pupil'}) 
    else:
        raise Exception(f'Do not know model label {model_label}')

    model.build_estimation_model()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('session', nargs='?', default=None)
    parser.add_argument('--roi', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(args.model_label, args.session, bids_folder=args.bids_folder, roi=args.roi)
