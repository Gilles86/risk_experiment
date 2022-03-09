import argparse
import os.path as op
import nibabel as nb
from nilearn import surface
import numpy as np


def main(subject, bids_folder):

    fn = op.join(bids_folder, 'derivatives', 'freesurfer', 
            '{fs_subject}', 'surf', '{hemi}.{label}.mgz')

    if subject == 'fsaverage':
        for hemi in['lh', 'rh']:
            fs_subject = subject
            im = nb.load(fn.format(hemi=hemi, label='wang2015_atlas', fs_subject=fs_subject))
            prob_mask = surface.load_surf_data(fn.format(hemi=hemi, label='wang2015_atlas', fs_subject=fs_subject))
            # print(prob_mask, op.exists(prob_mask))
            label_mask = np.in1d(prob_mask, range(18, 24)).astype(np.int16)
            
            new_im = nb.MGHImage(label_mask, im.affine, im.header)
            new_im.to_filename(fn.format(hemi=hemi, label='wang15_ips', fs_subject=fs_subject))
    else:
        fs_subject = f'sub-{subject}'
        for hemi in['lh', 'rh']:
            im = nb.load(fn.format(hemi=hemi, label='wang15_fplbl', fs_subject=fs_subject))
            prob_mask = surface.load_surf_data(fn.format(hemi=hemi, label='wang15_fplbl', fs_subject=fs_subject))
            # print(prob_mask, op.exists(prob_mask))
            label_mask = (prob_mask[18:24] > 0.05).any(0).astype(np.int16)
            
            new_im = nb.MGHImage(label_mask, im.affine, im.header)
            new_im.to_filename(fn.format(hemi=hemi, label='wang15_ips', fs_subject=fs_subject))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.bids_folder, )
