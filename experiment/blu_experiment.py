import argparse
from task import TaskSession
from calibrate import CalibrationSession
from task import TaskSession
from text import TextSession
from utils import run_experiment, fit_psychometric_curve, create_design, get_output_dir_str
from payout import get_payout
from psychopy import logging
import os.path as op
import os


def main(subject, session, settings, test=False, skip_calibration=False):

    output_dir, output_str = get_output_dir_str(subject, session, 'calibrate', None)

    if test:
        log_file = op.abspath('logs/test_log/sub-test_ses-test_task-calibrate_events.tsv')

    else:
        if not skip_calibration:
            calibrate_session = run_experiment(CalibrationSession, task='calibrate',
                                               subject=subject, session=session, settings=settings,
                                               use_runs=False)

    logging.warn(op.join(output_dir, output_str))
    log_file = op.join(output_dir, output_str + '_events.tsv')

    x_lower, x_upper = fit_psychometric_curve(log_file)

    logging.warn(f'Range: {x_lower}, {x_upper}')
    if test:
        limit = 1
    else:
        limit = None

    fn = make_trial_design(subject, x_lower, x_upper, limit=limit)
    logging.warn(fn)

    task_session = run_experiment(TaskSession, task='task', session=session, settings=settings, subject=subject)

    txt, payout = get_payout(subject)

    txt += '\nPlease remain seated until you get picked up for payment.'

    payout_folder = task_session.settings['output'].get('payout_folder')

    payout_fn = op.join(payout_folder, f'sub-{subject}_payout.txt')

    print(fn)

    with open(payout_fn, 'w') as f:
        f.write(str(payout))

    payout_session = TextSession(txt=txt,
            output_str='txt',
            output_dir=output_dir,
            settings_file=task_session.settings_file)

    payout_session.run()


def make_trial_design(subject, x_lower, x_upper, limit=None):
    import numpy as np

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

    df = create_design(prob1, prob2, fractions, repetitions=2)

    if limit is not None:
        df = df.iloc[:limit]

    df.to_csv(fn, sep='\t')
    logging.warning(f'Task settings written to {fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default='blu', nargs='?')
    parser.add_argument('--settings', default='blu', nargs='?')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--skip_calibration', action='store_true')
    args = parser.parse_args()

    main(subject=args.subject, session=args.session, settings=args.settings, test=args.test, skip_calibration=args.skip_calibration)
