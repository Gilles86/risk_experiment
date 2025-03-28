from session import NumeralRiskSession
import argparse
from utils import get_output_dir_str, get_settings



def main(subject, settings='blu'):

    output_dir, output_str = get_output_dir_str(subject, None, 'numeral_gambles', run=None)
    settings_fn = get_settings(settings)

    session = NumeralRiskSession(output_str=output_str, output_dir=output_dir, subject=subject, settings_file=settings_fn, run=None)

    session.create_trials()

    session.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the experiment.')
    parser.add_argument('subject', type=str, help='Subject identifier')
    parser.add_argument('--settings', type=str, help='Settings label', default='blu')
    args = parser.parse_args()

    main(args.subject, settings=args.settings)