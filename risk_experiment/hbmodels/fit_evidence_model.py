import os
import os.path as op
import argparse
from risk_experiment.utils.data import get_all_behavior
from evidence_model import EvidenceModel, EvidenceModelRegression, EvidenceModelSinglePrior
import arviz as az
import re
import pandas as pd
from scipy.stats import zscore
import numpy as np

def main(model_label, session='7t2', bids_folder='/data'):

    if model_label not in ['model1', 'certainty', 'certainty_full', 'singleprior']:
        raise NotImplementedError(f'Not implemented {model_label}')

    df = get_all_behavior(sessions=session, bids_folder=bids_folder)
    print(df)
    df = df.drop(3).drop(32)
    
    if model_label == 'model1':
        model = EvidenceModel(df)
    elif model_label == 'singleprior':
        model = EvidenceModelSinglePrior(df)
    elif model_label.startswith('certainty'):
        from scipy.stats import zscore

        df['z_certainty'] = df.groupby(['subject']).certainty.apply(zscore)
        df['z_certainty'] = df['z_certainty'].fillna(0.0)

        if model_label == 'certainty':
            model = EvidenceModelRegression(df, regressors={'evidence_sd1':'z_certainty',
                                                            'evidence_sd2':'z_certainty'})
        elif model_label == 'certainty_full':
            model = EvidenceModelRegression(df, regressors={'evidence_sd1':'z_certainty',
                                                            'evidence_sd2':'z_certainty',
                                                            'risky_prior_mu':'z_certainty',
                                                            'risky_prior_sd':'z_certainty'})

    elif model_label.startswith('subcortex'):
        reg = re.compile('subcortex_(?P<mask>lcR|lcL|vta|snc|lc)')
        mask = reg.match(model_label).group(1)
        print(mask)

        def get_signal(mask):
            signal = []
            for (subject, session), d in df.groupby(['subject', 'session']):
                ts = op.join(bids_folder, 'derivatives', 'extracted_signal', 
                f'sub-{subject:02d}', f'ses-{session}', 'func', f'sub-{subject:02d}_ses-{session}_desc-{mask}.singletrial_timeseries.tsv')
                ts = pd.read_csv(ts, sep='\t', index_col=['subject', 'session', 'trial_nr'])
                signal.append(ts)
            return pd.concat(signal)

        if mask not in ['lc']:
            signal = get_signal(mask)
        else:
            signal = (get_signal(mask+'L')[mask+'L'] + get_signal(mask+'R')[mask+'R']).to_frame(mask)

        signal = np.clip(signal, -10, 10)
        signal = signal.groupby(['subject', 'session']).apply(zscore)
        print(signal)
        print(df.join(signal))
        df = df.join(signal)
 
        model = EvidenceModelRegression(df, regressors={'evidence_sd1':mask})
    else:
        raise NotImplementedError(f'Not implemented {model_label}')

    model.build_model()
    trace = model.sample(1000, 1000)

    target_folder = op.join(bids_folder, 'derivatives', 'evidence_models')
    if not op.exists(target_folder):
        os.makedirs(target_folder)
        
    az.to_netcdf(trace,  op.join(target_folder, f'evidence_ses-{session}_model-{model_label}.nc'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default=None)
    parser.add_argument('session', default='7t2')
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.model, args.session, bids_folder=args.bids_folder)
