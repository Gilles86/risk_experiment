import glob
import argparse
import pandas as pd
import os.path as op
import numpy as np
import re
from text import TextSession
from utils import get_output_dir_str


def get_payout(subject, session, settings='default'):

    task = 'payout'

    logs_calibrate = glob.glob(op.abspath(op.join(
        'logs', f'sub-{subject}', f'ses-{session}', f'sub-{subject}_ses-{session}_task-calibration_run-*_events.tsv')))
    print(op.abspath(op.join('logs', f'sub-{subject}', f'ses-{session}',
                             f'sub-{subject}_ses-{session}_task-calibrate_run-*_events.tsv')))
    print(logs_calibrate)
    logs_task = glob.glob(op.abspath(op.join(
        'logs', f'sub-{subject}', f'ses-{session}', f'sub-{subject}_ses-{session}_task-task_run-*_events.tsv')))

    reg = re.compile(
        '.*sub-(?P<subject>.+)_ses-(?P<session>.+)_task-(?P<task>.+)_run-(?P<run>[0-9]+)_events\.tsv')
    df = []

    for l in (logs_calibrate + logs_task):
        d = pd.read_table(l)
        for key, value in reg.match(l).groupdict().items():
            d[key] = value

        df.append(d)

    df = pd.concat(df)
    df = df[np.in1d(df.phase, [8, 9])]
    df = df.pivot_table(index=['task', 'trial_nr'], values=[
                        'choice', 'certainty', 'n1', 'n2', 'prob1', 'prob2'])

    row = df.sample().iloc[0]

    txt = 'The stimulus computer randomly selected one of the choices ' \
    'you made in this session\n\n'

    if np.isnan(row.choice):
        txt += f'On the selected trial, you gave NO answer. '\
            'This means you will not get a bonus'
        payout = 0
    else:
        txt += f'You chose between {int(row.prob1*100):d}% probability of ' \
            f'winning {int(row.n1)} CHF, or {int(row.prob2*100):d}% probability ' \
            f'of winning {int(row.n2)} CHF.'

        if ((row.choice == 1) and (row['prob1'] == 1)):
            txt += f'\n\nYou chose the safe option and get {int(row.n1)} CHF'
            payout = row.n1

        if ((row.choice == 2) and (row.prob2 == 1)):
            txt += f'\n\nYou chose the safe option and get {int(row.n2)} CHF'
            payout = row.n2

        if ((row.choice == 2) and (row.prob1 == 1)):
            txt += '\n\nYou chose the risky option (pile 2). '
            die = np.random.randint(100) + 1
            txt += f'The die turned out to be {die}. '
            if die > 55:
                txt += '\n\nSo you got NO bonus.'
                payout = 0
            else:
                txt += f'\n\nSo you got a bonus of {int(row.n2)} CHF. '
                payout = row.n2

        if ((row.choice == 1) and (row.prob2 == 1)):
            txt += '\n\nYou chose the risky option (pile 1). '
            die = np.random.randint(100) + 1
            txt += f'The die turned out to be {die}. '
            if die > 55:
                txt += f'\n\nSo you got NO bonus'
                payout = 0
            else:
                txt += f'\n\nSo you got a bonus of {int(row.n1)} CHF.'
                payout = row.n1

    txt += '\n\nNote that after four sessions you will get the average bonus over the four sessions.'

    output_dir, output_str = get_output_dir_str(subject, session, task, None)

    settings_fn = op.join(op.dirname(__file__), 'settings',
                          f'{settings}.yml')

    payout_session = TextSession(txt=txt,
                                 output_str='payout',
                                 output_dir=output_dir,
                                 settings_file=settings_fn)

    payout_session.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    parser.add_argument('--settings', default='default', nargs='?')

    args = parser.parse_args()

    get_payout(subject=args.subject, session=args.session,
               settings=args.settings)
