import os.path as op
import argparse
import numpy as np
import scipy.stats as ss
import pandas as pd
from psychopy import logging
from itertools import product


def run_experiment(session_cls, task, use_runs=False, *args, **kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    parser.add_argument('run', default=None, nargs='?')
    parser.add_argument('--settings', default='default', nargs='?')
    args = parser.parse_args()

    if args.subject is None:
        subject = input('Subject? (999): ')
        subject = 999 if subject == '' else subject
    else:
        subject = args.subject

    if args.subject is None:
        session = input('Session? (1): ')
        session = 1 if session == '' else session
    else:
        session = args.session

    if args.run is None:
        run = input('Run? (None): ')
        run = None if run == '' else run
    elif args.run == '0':
        run = None
    else:
        run = args.run

    settings = op.join(op.dirname(__file__), 'settings',
                       f'{args.settings}.yml')
    logging.warn(f'Using {settings} as settings')
    output_dir = op.join(op.dirname(__file__), 'logs', f'sub-{subject}')
    logging.warn(f'Writing results to  {output_dir}')

    if session:
        output_dir = op.join(output_dir, f'ses-{session}')
        output_str = f'sub-{subject}_ses-{session}_task-{task}'
    else:
        output_str = f'sub-{subject}_task-{task}'

    if run:
        output_str += f'_run-{run}'

    log_file = op.join(output_dir, output_str + '_log.txt')

    if op.exists(log_file):
        overwrite = input(
            f'{log_file} already exists! Are you sure you want to continue? ')
        if overwrite != 'y':
            raise Exception('Run cancelled: file already exists')

    session = session_cls(output_str=output_str,
                          output_dir=output_dir,
                          settings_file=settings, subject=subject)
    session.create_trials()
    session.run()
    session.quit()


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
                  base=[5, 7, 10, 14, 20, 28], repetitions=1, n_runs=3):

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
