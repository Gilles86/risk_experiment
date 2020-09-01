import argparse
import pandas as pd
import numpy as np
from itertools import product
import os
import os.path as op
from utils import create_design


def main(subject, test=False):
    N_RUNS = 3

    h = np.arange(1, 9)
    fractions = 2**(h/4)
    if test:
        print("TEST")
        base = np.array([5, 10])
        prob1 = [1.]
        prob2 = [.55]
    else:
        base = np.array([5, 7, 10, 14, 20, 28])
        prob1 = [1., .55]
        prob2 = [.55, 1.]


    df = create_design(prob1, prob2, fractions, base=base)

    n_trials = len(df)

    calibrate_settings_folder = op.abspath(op.join('settings', 'calibration'))
    if not op.exists(calibrate_settings_folder):
        os.makedirs(calibrate_settings_folder)

    fn = op.abspath(op.join(calibrate_settings_folder,
                            f'sub-{subject}_ses-calibrate.tsv'))
    df.to_csv(fn, sep='\t')

    print(f'Wrote {n_trials} trials to {fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    main(subject=args.subject, test=args.test)
