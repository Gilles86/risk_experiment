import os.path as op
import argparse
import numpy as np
import scipy.stats as ss
import pandas as pd
from psychopy import logging
from itertools import product
import yaml

def run_experiment(session_cls, task, use_runs=False, subject=None, session=None, run=None, settings='default', n_runs=4, *args, **kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=subject, nargs='?')
    parser.add_argument('session', default=session, nargs='?')
    parser.add_argument('run', default=run, nargs='?')
    parser.add_argument('--settings', default=settings, nargs='?')
    parser.add_argument('--overwrite', action='store_true')
    cmd_args = parser.parse_args()
    subject, session, run, settings = cmd_args.subject, cmd_args.session, cmd_args.run, cmd_args.settings

    if subject is None:
        subject = input('Subject? (999): ')
        subject = 999 if subject == '' else subject

    if session is None:
        session = input('Session? (1): ')
        session = 1 if session == '' else session

    if use_runs and (run is None):
        run = input('Run? (None): ')
        run = None if run == '' else run
    elif run == '0':
        run = None

    settings_fn = op.join(op.dirname(__file__), 'settings',
                       f'{settings}.yml')

    with open(settings_fn, 'r') as f_in:
        settings_ = yaml.safe_load(f_in)

    if 'eyetracker' in settings_.keys():
        eyetracker_on = True
        logging.warn("Using eyetracker")
    else:
        eyetracker_on = False
        logging.warn("Using NO eyetracker")

    logging.warn(f'Using {settings_fn} as settings')


    if run is None:
        runs = range(1, n_runs + 1)
    else:
        runs = [run] 

    for run in runs:
        output_dir, output_str = get_output_dir_str(subject, session, task, run)

        log_file = op.join(output_dir, output_str + '_log.txt')
        logging.warn(f'Writing results to: {log_file}')

        if (not cmd_args.overwrite) and op.exists(log_file):
            overwrite = input(
                f'{log_file} already exists! Are you sure you want to continue? ')
            if overwrite != 'y':
                raise Exception('Run cancelled: file already exists') 
        session_object = session_cls(output_str=output_str,
                              output_dir=output_dir,
                              settings_file=settings_fn, subject=subject,
                              run=run,
                              eyetracker_on=eyetracker_on, *args, **kwargs)
        session_object.create_trials()
        logging.warn(f'Writing results to: {op.join(session_object.output_dir, session_object.output_str)}')
        session_object.run()
        session_object.close()

    return session


def sample_isis(n, s=1.0, loc=0.0, scale=10, cut=30):

    d = np.zeros(n, dtype=int)
    changes = ss.lognorm(s, loc, scale).rvs(n)
    changes = changes[changes < cut]

    ix = np.cumsum(changes).astype(int)
    ix = ix[ix < len(d)]
    d[ix] = 1

    return d


def create_stimulus_array_log_df(stimulus_arrays, index=None):

    stimuli = [pd.DataFrame(sa.xys, columns=['x', 'y'],
                            index=pd.Index(np.arange(1, len(sa.xys)+1), name='stimulus')) for sa in stimulus_arrays]

    stimuli = pd.concat(stimuli, ignore_index=True)

    if index is not None:
        stimuli.index = index

    return stimuli


def create_design(prob1, prob2, fractions,
                  base=[5, 7, 10, 14, 20, 28], repetitions=1, n_runs=4):

    base = np.array(base)
    repetition = range(1, repetitions+1)

    df = []
    for ix, p in enumerate(prob1):
        tmp = pd.DataFrame(product(base, fractions, repetition), columns=[
                           'base number', 'fraction', 'repetition'])
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
        (np.arange(len(p))+1) / (len(p) / n_runs))).astype(int)

    df = df.set_index(['run', 'p1'])
    ixs = np.random.permutation(df.index.unique())
    df = df.loc[ixs].sort_index(
        level='run', sort_remaining=False).reset_index('p1')

    df['trial'] = np.arange(1, len(df)+1)

    return df


def fit_psychometric_curve(log_file, plot=False, thresholds=(1, 4)):
    import statsmodels.api as sm
    df = pd.read_table(log_file)

    df = df[df.phase == 9]
    df = df.pivot_table(index=['trial_nr'], values=['choice', 'certainty', 'n1', 'n2', 'prob1', 'prob2'])
    df = df[~df.choice.isnull()]

    df['log(risky/safe)'] = np.log(df['n1'] / df['n2'])
    ix = df.prob1 == 1.0

    print(df)

    if ix.sum() > 0:
        df.loc[ix, 'log(risky/safe)'] = np.log(df.loc[ix, 'n2'] / df.loc[ix, 'n1'])
        df.loc[ix, 'chose risky'] = df.loc[ix, 'choice'] == 2

    if (~ix).sum() > 0:
        df.loc[~ix, 'log(risky/safe)'] = np.log(df.loc[~ix, 'n1'] / df.loc[~ix, 'n2'])
        df.loc[~ix, 'chose risky'] = df.loc[~ix, 'choice'] == 1

    df['chose risky'] = df['chose risky'].astype(bool)

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fac = sns.lmplot('log(risky/safe)', 'chose risky', data=df, logistic=True)

        for color, x in zip(sns.color_palette()[:4], [np.log(1./.55)]):
            
            plt.axvline(x, color=color, ls='--')    
            
        plt.gcf().set_size_inches(14, 6)
        plt.axhline(.5, c='k', ls='--')
        x = np.linspace(0, 1.5, 17)
        plt.xticks(x, [f'{e:0.2f}' for e in np.exp(x)], rotation='vertical')
        plt.xlim(0, 1.5)
        plt.show()

    
    # Fit probit
    df['intercept'] = 1

    try:
        m = sm.Probit(df['chose risky'], df[['intercept', 'log(risky/safe)']])
        r = m.fit()
        x_lower = (ss.norm.ppf(.2) - r.params.intercept) / r.params['log(risky/safe)']
        x_upper = (ss.norm.ppf(.8) - r.params.intercept) / r.params['log(risky/safe)']
    except Exception as e:
        print("Problem with calibration, using standard values")
        x_lower = np.log(thresholds[0])
        x_upper = np.log(thresholds[1])

    print(f'Original bounds: {np.exp(x_lower)}, {np.exp(x_upper)}')
    x_lower = np.exp(np.max((x_lower, np.log(thresholds[0]))))
    x_upper = np.exp(np.min((x_upper, np.log(thresholds[1]))))
    print(f'Final bounds: {x_lower}, {x_upper}')


    return x_lower, x_upper

def get_output_dir_str(subject, session, task, run):
    output_dir = op.join(op.dirname(__file__), 'logs', f'sub-{subject}')
    logging.warn(f'Writing results to  {output_dir}')

    if session:
        output_dir = op.join(output_dir, f'ses-{session}')
        output_str = f'sub-{subject}_ses-{session}_task-{task}'
    else:
        output_str = f'sub-{subject}_task-{task}'

    if run:
        output_str += f'_run-{run}'

    return output_dir, output_str
