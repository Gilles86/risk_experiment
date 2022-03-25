import argparse
import arviz as az
import os
import os.path as op
from risk_experiment.utils import get_task_paradigm
from tqdm.contrib.itertools import product
import pandas as pd
import numpy as np
import bambi as bmb

bids_folder = '/data'


models = {'model1': 'chose_risky ~ 1 + x*session + (x*session|subject)',
          'model2': 'chose_risky ~ 1 + x*risky_first + x*session + risky_first*session + (x*risky_first|subject) + (x*session|subject)'}


def main(model, bids_folder='/data'):

    df = get_task_paradigm(bids_folder=bids_folder)
    print(df.shape)
    df = df[~df.choice.isnull()]
    print(df.shape)

    df['x'] = df['log(risky/safe)']

    formula = models[model]

    model_probit = bmb.Model(formula, df[['chose_risky', 'x', 'risky_first']].reset_index(
    ), family="bernoulli", link='probit')

    results = model_probit.fit(2000, 2000, target_accept=.85)

    az.to_netcdf(results, op.join(bids_folder, 'derivatives',
                                  'probit_models', f'group_model-{model}_behavioralmodel.nc'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.model, bids_folder=args.bids_folder)
