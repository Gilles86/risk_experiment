import argparse
import pandas as pd
import numpy as np
from itertools import product
import os
import os.path as op


def main(subject):
    N_RUNS = 3

    h = np.arange(1, 9)
    fractions = 2**(h/4)
    base = np.array([5, 7, 10, 14, 20, 28])
    prob1 = [1., .45, .55, .67]
    prob2 = [.55, 1., 1., 1.]

    design = list(product(fractions, base, prob1))

    df = []
    for ix, p in enumerate(prob1):
        tmp = pd.DataFrame(product(base, fractions), columns=[
                           'base number', 'fraction'])
        tmp['p1'] = p
        tmp['p2'] = prob2[ix]

        df.append(tmp)

    df = pd.concat(df).reset_index(drop=True)

    df.loc[df['p1'] == 1.0, 'n1'] = df['base number']
    df.loc[df['p1'] != 1.0, 'n2'] = df['base number']
    df.loc[df['p1'] == 1.0, 'n2'] = (
        df['fraction'] * df['base number']).astype(int)
    df.loc[df['p1'] != 1.0, 'n1'] = (
        df['fraction'] * df['base number']).astype(int)

    df['n1'] = df['n1'].astype(int)
    df['n2'] = df['n2'].astype(int)

    # Shuffle _within_ p1's
    df = df.groupby('p1', as_index=False).apply(
        lambda d: d.sample(frac=1)).reset_index(level=0, drop=True)

    # Get run numbers
    df['run'] = df.groupby('p1').p1.transform(lambda p: np.ceil(
        (np.arange(len(p))+1) / (len(p) / N_RUNS))).astype(int)

    df = df.set_index(['run', 'p1'])
    ixs = np.random.permutation(df.index.unique())
    df = df.loc[ixs].sort_index(
        level='run', sort_remaining=False).reset_index('p1')

    calibrate_settings_folder = op.abspath(op.join('settings', 'calibration'))
    if not op.exists(calibrate_settings_folder):
        os.makedirs(calibrate_settings_folder)

    df['trial'] = np.arange(1, len(df)+1)
    df.to_csv(op.abspath(op.join(calibrate_settings_folder,
                                 f'sub-{subject}_ses-calibrate.tsv')), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    args = parser.parse_args()

    main(subject=args.subject)
