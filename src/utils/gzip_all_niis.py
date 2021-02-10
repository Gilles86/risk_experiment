import argparse
import os
import os.path as op
import glob
from pigz_python import compress_file


def main(subject, session, bids_folder):

    source_folder = op.join(bids_folder, 'sourcedata', f'sub-{subject}',
            'mri', f'ses-{session}')

    niis = glob.glob(op.join(source_folder, '*.nii'))

    print(niis)

    for nii in niis:
        print(nii)
        os.chdir(source_folder)
        compress_file(nii)
        os.remove(nii)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder)
