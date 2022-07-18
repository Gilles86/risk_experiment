import argparse
import numpy as np
import os
import nibabel.freesurfer.mghformat as fsmgh
import os.path as op
import neuropythy
from neuropythy.freesurfer import subject
from neuropythy.io import load, save
from neuropythy.mri import (is_image, is_image_spec, image_clear, to_image)


def main(subj, bids_folder='/data'):

    def read_surf_file(flnm):
      if flnm.endswith(".mgh") or flnm.endswith(".mgz"):
        data = np.array(fsmgh.load(flnm).dataobj).flatten()
      else:
        data = fsio.read_morph_data(flnm)
      return data

    target_dir = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subj}',
            'anat')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    target_fn = op.join(target_dir, f'sub-{subj}_space-T1w_desc-wang15ips_mask.nii.gz')

    sub = subject(op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subj}'))

    im = load(op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subj}', 'anat', f'sub-{subj}_desc-preproc_T1w.nii.gz'))
    im = to_image(image_clear(im, fill=0.0), dtype=np.int)

    mask_l = op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subj}', 'surf', 'lh.wang15_ips.mgz')
    mask_r = op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subj}', 'surf', 'rh.wang15_ips.mgz')

    lhdat = read_surf_file(mask_l)
    rhdat = read_surf_file(mask_r)

    print(lhdat.sum())
    print(rhdat.sum())

    print('Generating volume...')
    new_im = sub.cortex_to_image((lhdat, rhdat),
            im,
            hemi=None,
            method='nearest',
            fill=0.0)

    print('Exporting volume file: %s' % target_fn)
    save(target_fn, new_im)
    print('surface_to_image complete!')

    target_fn = op.join(target_dir, f'sub-{subj}_space-T1w_desc-wang15ipsL_mask.nii.gz')
    print('Generating volume...')
    new_im = sub.cortex_to_image(lhdat,
            im,
            hemi='lh',
            method='nearest',
            fill=0.0)
    print('Exporting volume file: %s' % target_fn)
    save(target_fn, new_im)
    print('surface_to_image complete!')

    target_fn = op.join(target_dir, f'sub-{subj}_space-T1w_desc-wang15ipsR_mask.nii.gz')
    print('Generating volume...')

    # note: somehow hemi='rh' does not work here
    new_im = sub.cortex_to_image((np.zeros_like(lhdat), rhdat),
            im,
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
