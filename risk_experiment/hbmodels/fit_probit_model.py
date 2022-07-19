import os
import os.path as op
import argparse
from risk_experiment.utils.data import get_all_behavior
from probit_model import ProbitModel
import arviz as az


def main(model_type, session='7t2', bids_folder='/data'):

    if model_type not in ['model1', 'model2']:
        raise NotImplementedError(f'Not implemented {model_label}')

    df = get_all_behavior(sessions=session, bids_folder=bids_folder)
    model = ProbitModel(df, model_type, bids_folder)
    model.build_model()

    trace = model.sample(500, 500)

    target_folder = op.join(bids_folder, 'derivatives', 'probit_models')
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
