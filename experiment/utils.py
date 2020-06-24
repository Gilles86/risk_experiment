import os.path as op
import argparse

def run_experiment(session_cls, name, *args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    args = parser.parse_args()
    
    if args.subject is None:
        subject = input('Subject? (999): ')
        subject = 999 if subject  == '' else subject
    else:
        subject = args.subject

    if args.subject is None:
        session = input('Session? (1): ')
        session = 1 if session  == '' else session
    else:
        session = args.session

    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = session_cls(f'sub-{subject}_ses-{name}_{session}', settings_file=settings)
    session.create_trials()
    session.run()
    session.quit()
