import glob
import pandas as pd
import os
import os.path as op
import argparse
import re
import shutil
from warnings import warn
import json
from nilearn import image
import numpy as np
from itertools import product
from tqdm import tqdm
from risk_experiment.utils.data import get_all_subjects, get_all_sessions


def main(subject, session, bids_folder):

    fns = glob.glob(op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'fmap',
                            f'*.json'))

    for fn in fns:
        with open(fn) as handle:
            metadata = json.load(handle)

            if metadata['IntendedFor'].startswith('func'):
                metadata['IntendedFor'] = metadata['IntendedFor'].replace('func/', f'ses-{session}/func/')
                print(metadata)
        
        with open(fn, 'w') as handle:
            json.dump(metadata, handle, indent=4)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', nargs='?', default=None)
    parser.add_argument('session', nargs='?', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    parser.add_argument(
        '--run_all', action='store_true')
    args = parser.parse_args()

    if args.run_all:
        for subject, session in product(get_all_subjects(), get_all_sessions()):
            main(subject, session, args.bids_folder)

    else:
        main(args.subject, args.session, args.bids_folder)
