import argparse
import arviz as az
import os
import os.path as op
from risk_experiment.utils import get_all_task_behavior
from tqdm.contrib.itertools import product
import pandas as pd
import numpy as np
import bambi as bmb
from scipy.stats import zscore

bids_folder = '/data'


models = {'model1': 'chose_risky ~ 1 + x*session + (x*session|subject)',
          'model5': 'chose_risky ~ 1 + x*session + (x*session|subject)',
          'model2': 'chose_risky ~ 1 + x*risky_first + x*session + risky_first*session + (x*risky_first|subject) + (x*session|subject)',
          'model3': 'chose_risky ~ 1 + x*session + x*C(certainty) + (x*session|subject) + (x*C(certainty)|subject)',
          'model4': 'chose_risky ~ 1 + x*session + x*certainty + (x*session|subject) + (x*certainty|subject)',
          'model6': 'chose_risky ~ 1 + C(base_number)*x + x*session + (C(base_number)*x|subject) + (x*session|subject)',
          'model7': 'chose_risky ~ 1 + C(base_number):x + x*session + (C(base_number):x|subject) + (x*session|subject)',
          'model8': 'chose_risky ~ 1 + C(base_number):x:risky_first + C(base_number):x + x*session + (C(base_number):x:risky_first|subject) + (C(base_number):x|subject) + (x*session|subject)',
          'model9': 'chose_risky ~ 1 + C(base_number):risky_first + C(base_number):x:risky_first + C(base_number):x + x*session + (C(base_number):x:risky_first|subject) + (C(base_number):x|subject) + (x*session|subject) + (C(base_number):risky_first|subject)'}


def main(model, bids_folder='/data'):

    # df = get_task_paradigm(bids_folder=bids_folder)
    df = get_all_task_behavior(bids_folder=bids_folder)
    print(df.shape)
    df = df[~df.choice.isnull()]
    print(df.shape)

    df['x'] = df['log(risky/safe)']
    df.loc[df.risky_first, 'base_number'] = df['n2']
    df.loc[~df.risky_first, 'base_number'] = df['n1']

    formula = models[model]

    print(df)

    if model != 'model1':
        df = df.drop(['03', '32'], level='subject')

    print(df)

    if model == 'model3':
        df = df[~df.certainty.isnull()]

    if model == 'model4':
        df = df[~df.certainty.isnull()]
        df['certainty'] = df['certainty'].astype(float)
        df['certainty'] = df['certainty'].groupby(['subject', 'session']).apply(zscore)
        print(df)
        df = df[~df.certainty.isnull()]
        print(df)


    model_probit = bmb.Model(formula, df[['chose_risky', 'x', 'risky_first', 'certainty', 'base_number']].reset_index(
    ), family="bernoulli", link='probit')
    print(model_probit)

    results = model_probit.fit(2000, 2000, target_accept=.85, init='adapt_diag')

    az.to_netcdf(results, op.join(bids_folder, 'derivatives',
                                  'probit_models', f'group_model-{model}_behavioralmodel.nc'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.model, bids_folder=args.bids_folder)
