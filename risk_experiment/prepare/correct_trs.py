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

    fns = glob.glob(op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'func',
                            f'sub-{subject}_ses-{session}_*.nii*'))
    print(fns)

    for fn in tqdm(fns):
        im = image.load_img(fn)

        zooms = im.header.get_zooms()

        if abs(zooms[-1] - 2.3) > .2:
            print('yo')
            zooms = list(zooms)
            if session[:2] == '7t':
                zooms[-1] = 2.3
            elif sesion[:2] == '3t':
                zooms[-1] = 2.298
            im.header.set_zooms(zooms)
            im.to_filename(fn)

        print(im.header.get_zooms())


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
            if int(subject) > 26:
                main(subject, session, args.bids_folder)

    else:
        main(args.subject, args.session, args.bids_folder)
