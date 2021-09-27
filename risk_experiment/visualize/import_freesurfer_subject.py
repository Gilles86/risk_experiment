import argparse
from cortex import freesurfer
import os.path as op


def main(subject, bids_folder):

    subject = int(subject)

    freesurfer.import_subj(f'sub-{subject:02d}', 
            freesurfer_subject_dir=op.join(bids_folder, 'derivatives', 'freesurfer'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()
    main(args.subject, args.bids_folder)
