import os
import os.path as op
import arviz as az
import argparse
from risk_experiment.utils.data import get_all_behavior
from fit_model import build_model
from tms_risk.cogmodels.utils import plot_ppc, cluster_offers

def main(model_label, session, bids_folder, group_only=False, subject=None, col_wrap=5, thin=5):

    df = get_all_behavior(bids_folder=bids_folder)
    df = df.xs(session, 0, 'session')

    trace_folder = op.join(bids_folder, 'derivatives', 'cogmodels')
    trace = az.from_netcdf(
        op.join(trace_folder, f'ses-{session}_model-{model_label}_trace.netcdf'))

    df['log(risky/safe)'] = df.groupby(['subject'],
                                       group_keys=False).apply(cluster_offers)

    if thin is not None: 
        trace = trace.sel(draw=slice(None, None, thin))

    if group_only:
        levels = ['group']
    else:
        levels = ['group', 'subject']

    model = build_model(model_label, df)

    ppc = model.ppc(trace=trace, data=df)

    if subject is not None:
        df = df.xs(int(subject), 0, 'subject', drop_level=False)
        ppc = ppc.xs(int(subject), 0, 'subject', drop_level=False)
        levels = ['subject']
        col_wrap = 1

    for plot_type in [1,2,3,5]:
        for var_name in ['p', 'll_bernoulli']:
            for level in levels:
                target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'plots', level, var_name)

                if not op.exists(target_folder):
                    os.makedirs(target_folder)

                if subject is None:
                    fn = f'ses-{session}_plot-{plot_type}_model-{model_label}_pred.pdf'
                else:
                    fn = f'{session}_plot-{plot_type}_model-{model_label}_subject-{subject}_pred.pdf'

                plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name, col_wrap=col_wrap).savefig(
                    op.join(target_folder, fn))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    parser.add_argument('--group_only', action='store_true')
    parser.add_argument('--subject', default=None)
    args = parser.parse_args()

    main(args.model_label, args.session, bids_folder=args.bids_folder, group_only=args.group_only, subject=args.subject)