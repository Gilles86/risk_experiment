import os
import os.path as op
import arviz as az
from risk_experiment.utils.data import get_all_behavior
import argparse
import bambi
import pandas as pd
import scipy.stats as ss
import numpy as np


def main(model_label, session, bids_folder='/data/ds-risk'):

    model = build_model(model_label, session, bids_folder)

    idata = model.fit(init='adapt_diag',
    target_accept=0.9, draws=500, tune=500)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    az.to_netcdf(idata,
                 op.join(target_folder, f'ses-{session}_model-probit{model_label}_trace.netcdf'))

def build_model(model_label, session, bids_folder='/data/ds-risk'):

    model_label = int(model_label)

    df = get_all_behavior(bids_folder=bids_folder)
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

def invprobit(x):
    return ss.norm.ppf(x)

def extract_rnp_precision(trace, model, data, group=False):

    data = data.reset_index()

    if group:
        fake_data = pd.MultiIndex.from_product([data.reset_index()['subject'].unique()[[0]],
                                                [0, 1],
                                                data['n_safe'].unique(),
                                                [False, True]],
                                                names=['subject', 'x', 'n_safe', 'risky_first']
                                                ).to_frame().reset_index(drop=True)
    else:
        fake_data = pd.MultiIndex.from_product([data.reset_index()['subject'].unique(),
                                                [0, 1],
                                                data['n_safe'].unique(),
                                                [False, True]],
                                                names=['subject', 'x', 'n_safe', 'risky_first']
                                                ).to_frame().reset_index(drop=True)

    pred = model.predict(trace, 'mean', fake_data, inplace=False)['posterior']['chose_risky_mean']

    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake_data))

    # return pred

    pred0 = pred.xs(0, 0, 'x')
    intercept = pd.DataFrame(invprobit(pred0), index=pred0.index, columns=pred0.columns)
    gamma = invprobit(pred.xs(1, 0, 'x')) - intercept
    gamma = gamma.stack([-1, -2]).iloc[:, 0]
    rnp = np.exp(intercept/gamma)

    return pd.concat((intercept, gamma, rnp), axis=1, keys=['intercept', 'gamma', 'rnp'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(args.model_label, args.session, bids_folder=args.bids_folder)
