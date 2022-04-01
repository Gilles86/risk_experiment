import argparse
from nipype.interfaces.ants import ApplyTransforms
import os.path as op
import os

from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

bids_folder = '/data'
subject = '32'
mask = 'lcL'


def main(subject, session, mask, bids_folder):

    if session is None:
        target_dir = op.join(bids_folder, 'derivatives', 'masks', f'sub-{subject}', 'anat')
    else:
        target_dir = op.join(bids_folder, 'derivatives', 'masks', f'sub-{subject}', f'ses-{session}', 'anat')

        if session.endswith('2'):
            task = 'task'
        else:
            task = 'mapper'

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    applier = ApplyTransforms(interpolation='Linear')
    applier.inputs.input_image = op.join(bids_folder, 'derivatives', 'masks', f'group_space-MNI152NLin2009cAsym_desc-{mask}_mask.nii.gz')
    applier.inputs.transforms = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'anat',
                                        f'sub-{subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')

    if session is None:
        applier.inputs.reference_image = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'anat',
                                        f'sub-{subject}_desc-preproc_T1w.nii.gz')
        applier.inputs.output_image = op.join(target_dir, f'sub-{subject}_space-T1w_desc-{mask}_mask.nii.gz')
    else:
        applier.inputs.reference_image = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', f'ses-{session}',
                                        'func', f'sub-{subject}_ses-{session}_task-{task}_run-1_space-T1w_boldref.nii.gz')

        applier.inputs.output_image = op.join(target_dir, f'sub-{subject}_ses-{session}_space-T1w_desc-{mask}_mask.nii.gz')


    r = applier.run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('session', nargs='?', default=None)
    parser.add_argument('--mask', nargs='?', default='lcL')
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()
    print(args)

    main(args.subject, args.session, args.mask,  args.bids_folder)
