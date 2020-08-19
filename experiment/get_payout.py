import argparse
import pandas as pd
import os.path as op
import numpy as np

def main(subject):

    log_calibrate = op.abspath(op.join('logs', f'sub-{subject}', 'ses-blu',
                            f'sub-{subject}_ses-blu_task-calibrate_events.tsv'))
    log_calibrate = pd.read_table(log_calibrate)

    log_task = op.abspath(op.join('logs', f'sub-{subject}', 'ses-blu',
                            f'sub-{subject}_ses-blu_task-task_events.tsv'))
    log_task = pd.read_table(log_task)

    df = pd.concat((log_calibrate, log_task))
    print(df)

    row = df.sample().iloc[0]
    row.choice = 1

    if np.isnan(row.choice):
        txt = f'On the selected trial, you gave NO answer. '\
        'This means you will not get a bonus'
    else:
        txt = f'''
        You chose between {int(row.prob1*100):d}% chance on {row.n1} CHF,
        or {int(row.prob2*100):d}% chance on {row.n2} CHF. You chose option
        {row.choice}.
        '''

    print(txt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')

    args = parser.parse_args()

    main(subject=args.subject)
