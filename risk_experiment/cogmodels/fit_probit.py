import os
import os.path as op
import arviz as az
from risk_experiment.utils.data import get_all_behavior
import argparse
import bambi


def main(model_label, session, bids_folder='/data/ds-tmsrisk'):

    model = build_model(model_label, session, bids_folder)

    idata = model.fit(init='adapt_diag',
    target_accept=0.9, draws=500, tune=500)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    az.to_netcdf(idata,
                 op.join(target_folder, f'ses-{session}_model-probit{model_label}_trace.netcdf'))

def build_model(model_label, session, bids_folder='/data/ds-tmsrisk'):

    model_label = int(model_label)

    df = get_all_behavior(bids_folder=bids_folder)
    print(df)
    df = df.xs(session, 0, 'session')

    df['x'] = df['log(risky/safe)']
    df['chose_risky'] = df['chose_risky'].astype(bool)

    df = df.reset_index()

    if model_label == 1:
        formula = 'chose_risky ~ x*C(risky_first)*C(n_safe) + (x*C(risky_first)*C(n_safe)|subject)'
    elif model_label == 2:
        formula = 'chose_risky ~ x + (x|subject)'
    elif model_label == 3:
        formula = 'chose_risky ~ x*C(risky_first) + (x*C(risky_first)|subject)'

    return bambi.Model(formula, data=df, link='probit', family='bernoulli')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(args.model_label, args.session, bids_folder=args.bids_folder)
