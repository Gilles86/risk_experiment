import argparse
from task import TaskSession
from calibrate import CalibrationSession
from utils import run_experiment


def main(subject, session):
    run_experiment(CalibrationSession, task='calibrate',
                   subject=subject, session=session)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    args = parser.parse_args()

    main(subject=args.subject, session=args.session)
