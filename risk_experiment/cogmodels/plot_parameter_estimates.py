import os
import os.path as op
import argparse
import arviz as az
from risk_experiment.utils.data import get_all_behavior
from tms_risk.cogmodels.utils import plot_subjectwise_posterior, plot_groupwise_posterior , cluster_offers
from fit_model import build_model
import numpy as np

def main(model_label, session, bids_folder, group_only, thin=5):
    df = get_all_behavior(bids_folder=bids_folder)
    df = df.xs(session, 0, 'session')

    df['log(risky/safe)'] = df.groupby(['subject'],
                                       group_keys=False).apply(cluster_offers)

    trace_folder = op.join(bids_folder, 'derivatives', 'cogmodels')
    trace = az.from_netcdf(
        op.join(trace_folder, f'ses-{session}_model-{model_label}_trace.netcdf'))

    model = build_model(model_label, df)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'parameter_plots')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    for key in model.free_parameters:
        print(key)
        hue = None
        ref_value = None

        if 'evidence_sd' in key:
            hue = 'presentation'

        if key == 'prior_mu':
            ref_value = np.log(df['n1']).groupby('subject').mean().to_frame('ref_value')
            print(ref_value)

        if key == 'prior_sd':
            ref_value = np.log(df['n1']).groupby('subject').std().to_frame('ref_value')
            print(ref_value)

        if key == 'risky_prior_mu':
            ref_value = np.log(df['n_risky']).groupby('subject').mean().to_frame('ref_value')
            print(ref_value)

        if key == 'risky_prior_sd':
            ref_value = np.log(df['n_risky']).groupby('subject').std().to_frame('ref_value')
            print(ref_value)

        if key == 'alpha':
            ref_value = df.index.unique('subject').to_frame().drop('subject', axis=1)
            ref_value['ref_value'] = 1.0

        fac = plot_subjectwise_posterior(trace, key=key, hue=hue, ref_value=ref_value)
        fac.savefig(op.join(target_folder, f'ses-{session}_model-{model_label}_{key}_subject.pdf'))

        if ref_value is not None:
            ref_value = ref_value.mean()

        fac = plot_groupwise_posterior(trace, key=key, hue=hue, ref_value=ref_value)
        fac.savefig(op.join(target_folder, f'ses-{session}_model-{model_label}_{key}_group.pdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    parser.add_argument('--group_only', action='store_true')
    args = parser.parse_args()

    main(args.model_label, args.session, bids_folder=args.bids_folder, group_only=args.group_only)
