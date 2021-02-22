import argparse
from importlib import import_module
import pandas as pd
import re

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')

            #Allow for splitting lists
            if ',' in value:
                value = value.split(',')

            getattr(namespace, self.dest)[key] = value


def main(session_list, script, bids_folder, **kwargs):

    reg = re.compile('sub-(?P<subject>[a-zA-Z0-9]+)')

    if reg.match(session_list):
        subject = reg.match(session_list).group(1)
        sessions = pd.DataFrame([dict(subject=subject, session=s) for s in ['3t1', '3t2', '7t1', '7t2']])
    else:
        sessions = pd.read_csv(session_list, dtype={
                               'subject': str, 'session': str})

    print(sessions)

    mod = import_module(script, 'main_package')

    for ix, row in sessions.iterrows():
        print(row)
        try:
            mod.main(row.subject, row.session, bids_folder=bids_folder, **kwargs)
        except Exception as e:
            print(
                f"Problem with subject {row.subject}, session {row.session}:\n{e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_list', default=None)
    parser.add_argument('script', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    if args.kwargs is None:
        args.kwargs = {}

    main(args.subject_list, args.script, args.bids_folder, **args.kwargs)
