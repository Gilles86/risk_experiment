import os
import os.path as op
import argparse
from nipype.interfaces.freesurfer import SurfaceTransform
import nipype.pipeline.engine as pe
from nilearn import surface
from neuropythy.freesurfer import subject as fs_subject
from neuropythy.io import load, save
from neuropythy.mri import (is_image, is_image_spec, image_clear, to_image)
import numpy as np

def main(subject, bids_folder):

    fsnative_fn = op.join(bids_folder, 'derivatives', 'npc_com', f'sub-{subject}_space-fsnative-npcr_hemi-R_com.gii')
    mask_data = surface.load_surf_data(fsnative_fn).astype(bool)

    subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')

    target_dir = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}',
            'anat')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    target_fn = op.join(target_dir, f'sub-{subject}_space-T1w_desc-npccom_mask.nii.gz')
    sub = fs_subject(op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subject}'))
    im = load(op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'anat', f'sub-{subject}_desc-preproc_T1w.nii.gz'))
    im = to_image(image_clear(im, fill=0.0), dtype=np.int)

    print('Generating volume...')
    new_im = sub.cortex_to_image((np.zeros(sub.lh.vertex_count), mask_data),
            im,
            hemi=None,
            method='nearest',
            fill=0.0)

    print('Exporting volume file: %s' % target_fn)
    save(target_fn, new_im)
    print('surface_to_image complete!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.bids_folder, )
