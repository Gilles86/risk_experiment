from session import NumeralRiskSession
import argparse
from utils import get_output_dir_str, get_settings
from main import main as main_
import socket
import re
import argparse
import warnings
import os
import os.path as op


def copy_log_files(subject, in_blu=True):

    source_dir = op.abspath(op.join(op.dirname(__file__), 'logs', f'sub-{subject}'))
    
    if in_blu:
        destination_dir = op.join('N:', 'client_write', 'Gilles', 'logs',  f'sub-{subject}')
    else:
        destination_dir = op.join('/tmp', 'logs',  f'sub-{subject}')

    print(f'Copying log files from {source_dir} to {destination_dir}')

    if not op.exists(destination_dir):
        os.makedirs(destination_dir)
    for filename in os.listdir(source_dir):
        source_file_path = op.join(source_dir, filename)

        if os.path.isfile(source_file_path):
            destination_file_path = op.join(destination_dir, filename)

            with open(source_file_path, 'rb') as source_file:
                data = source_file.read()
 
            with open(destination_file_path, 'wb') as destination_file:
                destination_file.write(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the experiment.')
    parser.add_argument('session', type=int, help='Session number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    computer_name = socket.gethostname().lower()
    settings = 'blu'

    # econblulab00 - econblulab34
    match = re.search(r'econblulab(\d+)', computer_name)

    if match:
        subject = int(match.group(1))

        subject = subject + 1 + (args.session-1) * 35
        in_blu = True
    else:
        warnings.warn("No subject number found in the computer name. Using default subject number 0.")
        subject = -args.session
        in_blu = False

    main_(subject, settings=settings, debug=args.debug)

    copy_log_files(subject, in_blu)