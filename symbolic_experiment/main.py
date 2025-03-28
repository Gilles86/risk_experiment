from session import NumeralRiskSession
import argparse
from utils import get_output_dir_str, get_settings



def main(subject, settings='blu', debug=False):

    output_dir, output_str = get_output_dir_str(subject, None, 'numeral_gambles', run=None)
    settings_fn = get_settings(settings)


    session = NumeralRiskSession(output_str=output_str, output_dir=output_dir, subject=subject, settings_file=settings_fn, run=None)

    n_runs = 1 if debug else None
    n_trials_per_run = 1 if debug else None
    if debug:
        session.settings['task']['jitter1'] = [1.0]
        session.settings['task']['jitter2'] = [1.0]

    session.create_trials(n_runs=n_runs, n_trials_per_run=n_trials_per_run)

    session.run()
    print('yo1.5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the experiment.')
    parser.add_argument('subject', type=str, help='Subject identifier')
    parser.add_argument('--settings', type=str, help='Settings label', default='blu')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    main(args.subject, settings=args.settings, debug=args.debug)