import hedfpy
import os.path as op
import os
from risk_experiment.utils import run_main
from glob import glob


def main(subject, session, bids_folder):


    d_ = op.join(bids_folder, 'sourcedata', f'sub-{subject}', 'behavior', f'ses-{session}', )

    if session[-1] == '1':

        calibration_fn = op.join(d_, f'sub-{subject}_ses-{session}_task-calibration_run-1.edf')
        # fn = op.join(bids_folder, 'sourcedata', 
        # get_pupil_data(fn)
        print(d_)
        print(glob(op.join(d_, '*.edf')))

if __name__ == '__main__':
    run_main(main)
