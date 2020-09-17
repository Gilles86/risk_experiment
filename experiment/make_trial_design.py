import argparse
import os.path as op
import os
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
from utils import create_design
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

    print(log_file)

    df = pd.read_table(log_file)

    df = df[df.phase == 9]
    df = df.pivot_table(index=['trial_nr'], values=['choice', 'certainty', 'n1', 'n2', 'prob1', 'prob2'])

    df['log(risky/safe)'] = np.log(df['n1'] / df['n2'])
    ix = df.prob1 == 1.0

    df.loc[~ix, 'log(risky/safe)'] = np.log(df.loc[~ix, 'n1'] / df.loc[~ix, 'n2'])
    df.loc[ix, 'log(risky/safe)'] = np.log(df.loc[ix, 'n2'] / df.loc[ix, 'n1'])
    df.loc[~ix, 'chose risky'] = df.loc[~ix, 'choice'] == 1
    df.loc[ix, 'chose risky'] = df.loc[ix, 'choice'] == 2

    df['chose risky'] = df['chose risky'].astype(bool)

    #plot
    fac = sns.lmplot('log(risky/safe)', 'chose risky', data=df, logistic=True)

    for color, x in zip(sns.color_palette()[:4], [np.log(1./.55)]):
        
        plt.axvline(x, color=color, ls='--')    
        
    plt.gcf().set_size_inches(14, 6)
    plt.axhline(.5, c='k', ls='--')
    x = np.linspace(0, 1.5, 17)
    plt.xticks(x, [f'{e:0.2f}' for e in np.exp(x)], rotation='vertical')
    plt.xlim(0, 1.5)

    # Fit probit
    df['intercept'] = 1
    m = sm.Probit(df['chose risky'], df[['intercept', 'log(risky/safe)']])
    r = m.fit()

    print(r.params) 

    x_lower = (ss.norm.ppf(.2) - r.params.intercept) / r.params['log(risky/safe)']
    x_upper = (ss.norm.ppf(.8) - r.params.intercept) / r.params['log(risky/safe)']


    fractions = np.exp(np.linspace(x_lower, x_upper, 8))
    base = np.array([5, 7, 10, 14, 20, 28])
    prob1 = [1., .55]
    prob2 = [.55, 1.]
    repetition = (1, 2)

    task_settings_folder = op.abspath(op.join('settings', 'task'))
    if not op.exists(task_settings_folder):
        os.makedirs(task_settings_folder)

    fn = op.abspath(op.join(task_settings_folder,
                            f'sub-{subject}_ses-task.tsv'))

    df = create_design(prob1, prob2, fractions, repetitions=2)

    df.to_csv(fn, sep='\t')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    parser.add_argument('run', default=None, nargs='?')
    args = parser.parse_args()

    main(subject=args.subject, session=args.session, run=args.run)
