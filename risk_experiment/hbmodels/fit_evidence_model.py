import os
import os.path as op
import argparse
from risk_experiment.utils.data import get_all_behavior
from evidence_model import EvidenceModel, EvidenceModelRegression
import arviz as az


def main(model, session='7t2', bids_folder='/data'):

    if model not in ['model1', 'certainty']:
        raise NotImplementedError

    df = get_all_behavior(sessions=session, bids_folder=bids_folder)
    
    if model == 'model1':
        model = EvidenceModel(df)
    elif model == 'certainty':
        from scipy.stats import zscore

        df['z_certainty'] = df.groupby(['subject']).certainty.apply(zscore)
        df['z_certainty'] = df['z_certainty'].fillna(0.0)

        model = EvidenceModelRegression(df, regressors={'evidence_sd1':'z_certainty',
                                                        'evidence_sd2':'z_certainty'})

    model.build_model()
    trace = model.sample(500, 500)

    target_folder = op.join(bids_folder, 'derivatives', 'evidence_models')
    if not op.exists(target_folder):
        os.makedirs(target_folder)
        
    az.to_netcdf(trace,  op.join(target_folder, f'evidence_ses-{session}_model-{model}.nc'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default=None)
    parser.add_argument('session', default='7t2')
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.model, args.session, bids_folder=args.bids_folder)
