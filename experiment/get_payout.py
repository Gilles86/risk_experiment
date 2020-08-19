import argparse
import pandas as pd
import os.path as op
import numpy as np

def main(subject):

    log_calibrate = op.abspath(op.join('logs', f'sub-{subject}', 'ses-blu',
                            f'sub-{subject}_ses-blu_task-calibrate_events.tsv'))
    log_calibrate = pd.read_table(log_calibrate)
    log_calibrate['task'] = 'calibrate'

    log_task = op.abspath(op.join('logs', f'sub-{subject}', 'ses-blu',
                            f'sub-{subject}_ses-blu_task-task_events.tsv'))
    log_task = pd.read_table(log_task)
    log_task['task'] = 'task'

    df = pd.concat((log_calibrate, log_task))
    df = df[df.phase == 9]
    df = df.pivot_table(index=['task', 'trial_nr'], values=['choice', 'certainty', 'n1', 'n2', 'prob1', 'prob2'])
    print(df)

    row = df.sample().iloc[0]
    print(row)

    if np.isnan(row.choice):
        txt = f'On the selected trial, you gave NO answer. '\
        'This means you will not get a bonus'
    else:
        txt = f'''
You chose between {int(row.prob1*100):d}% chance on {row.n1} CHF,
or {int(row.prob2*100):d}% chance on {row.n2} CHF. You chose option
{row.choice}.'''

        if ((row.choice == 1) and (row['prob1'] == 1)):
            txt += '\nYou chose the safe option and get {row.n1} CHF'

        if ((row.choice == 2) and (row.prob2 == 1)):
            txt += '\nYou chose the safe option and get {row.n2} CHF'

        if ((row.choice == 2) and (row.prob1 == 1)):
            txt += '\n You chose the risky option (pile 2).'
            die = np.random.randint(100) + 1
            txt += 'The die turned out to be {die}'
            if die > 55:
                txt += 'So you got NO bonus'
            else:
                txt += 'So you got a bonus of {row.n2}'

        if ((row.choice == 1) and (row.prob2 == 1)):
            txt += '\n You chose the risky option (pile 1).'
            die = np.random.randint(100) + 1
            txt += f'The die turned out to be {die}. '
            if die > 55:
                txt += 'So you got NO bonus'
            else:
                txt += f'So you got a bonus of {row.n1} CHF.'

    print(txt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')

    args = parser.parse_args()

    main(subject=args.subject)
