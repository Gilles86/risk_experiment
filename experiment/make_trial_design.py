import argparse
import os.path as op
import os
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
from utils import create_design, fit_psychometric_curve
import matplotlib.pyplot as plt

def main(subject, session=None, run=None):

    log_file = op.abspath(op.join('logs', f'sub-{subject}'))

    if session:
    	log_file = op.join(log_file, f'ses-{session}')

    log_file = op.join(log_file, f'sub-{subject}')

    if session:
    	log_file += f'_ses-{session}'

    log_file += '_task-calibration'

    if run:
    	log_file += f'_run-{run}'

    log_file += '_events.tsv'

    x_lower, x_upper = fit_psychometric_curve(log_file, plot=True)

    print(log_file)

 

    print(x_lower, x_upper)
    fractions = np.exp(np.linspace(np.log(x_lower), np.log(x_upper), 8))
    base = np.array([5, 7, 10, 14, 20, 28])
    prob1 = [1., .55]
    prob2 = [.55, 1.]
    repetition = (1, 2)

    task_settings_folder = op.abspath(op.join('settings', 'task'))
    if not op.exists(task_settings_folder):
        os.makedirs(task_settings_folder)

    fn = op.abspath(op.join(task_settings_folder,
                            f'sub-{subject}_ses-task.tsv'))

    df = create_design(prob1, prob2, fractions, repetitions=2, n_runs=8)

    df.to_csv(fn, sep='\t')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    parser.add_argument('run', default=None, nargs='?')
    args = parser.parse_args()

    main(subject=args.subject, session=args.session, run=args.run)
