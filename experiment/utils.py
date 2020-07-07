import os.path as op
import argparse
import numpy as np
import scipy.stats as ss
import pandas as pd


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

    settings = op.join(op.dirname(__file__), 'settings', f'{args.settings}.yml')
    output_dir = op.join(op.dirname(__file__), 'logs', f'sub-{subject}')

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
