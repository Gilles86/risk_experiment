import argparse
from importlib import import_module
import pandas as pd


def main(session_list, script, bids_folder):

    sessions = pd.read_csv(session_list, dtype={
                           'subject': str, 'session': str})

    print(sessions)

    mod = import_module(script, 'main_package')

    for ix, row in sessions.iterrows():
        print(row)
        try:
            mod.main(row.subject, row.session, bids_folder=bids_folder)
        except Exception as e:
            print(
                f"Problem with subject {row.subject}, session {row.session}:\n{e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_list', default=None)
    parser.add_argument('script', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject_list, args.script, args.bids_folder)
