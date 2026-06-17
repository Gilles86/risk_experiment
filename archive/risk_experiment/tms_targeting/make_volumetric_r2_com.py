from nilearn import image
import os
import os.path as op
from scipy import ndimage
import numpy as np
import argparse
from nipype.interfaces import ants
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity

def main(subject, session, bids_folder):

    r2 = image.load_img(op.join(bids_folder, 'derivatives', 'encoding_model',
                                f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_desc-r2.optim_space-T1w_pars.nii.gz'))

    target_dir = op.join(bids_folder, 'derivatives', 'tms_targets', f'sub-{subject}', f'ses-{session}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    ips_mask1 = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-npc1R_mask.nii.gz')
    ips_mask2 = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-npc2R_mask.nii.gz')
    ips_mask = image.math_img('ips_mask1+ips_mask2', ips_mask2=ips_mask2, ips_mask1=ips_mask1)

    ips_mask.to_filename(op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-npc12R_mask.nii.gz'))
    ips_mask = image.resample_to_img(ips_mask, r2, interpolation='nearest')

    ips_mask = image.new_img_like(ips_mask, 
                                 ndimage.binary_dilation(ips_mask.get_fdata())[..., 0])

    thr_r2 = image.math_img('ips_mask*r2', r2=r2, ips_mask=ips_mask)

    def get_coords_ball(r2):
        vox_coords = np.unravel_index(r2.get_fdata().argmax(), r2.shape)
        coords = image.coord_transform(*vox_coords, thr_r2.affine)
        _, ball =_apply_mask_and_get_affinity([list(coords)], image.concat_imgs([r2]), 3., True)
        ball = ball.toarray().reshape(r2.shape)* 100.
        ball = image.new_img_like(r2, ball.astype(int))
        
        return coords, ball


    coords, ball = get_coords_ball(thr_r2)
    ball.to_filename(op.join(target_dir, f'sub-{subject}_desc-r2npc_mask.nii.gz'))

    r2_smooth = image.smooth_img(r2, 5.0)
    r2_smooth.to_filename(op.join(bids_folder, 'derivatives', 'encoding_model', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_desc-r2smooth.optim_space-T1w_pars.nii.gz'))

    thr_r2 = image.math_img('ips_mask*r2', r2=r2_smooth, ips_mask=ips_mask)
    coords, ball = get_coords_ball(thr_r2)
    ball.to_filename(op.join(target_dir, f'sub-{subject}_desc-r2smoothnpc_mask.nii.gz'))


    to_mni = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', f'anat', f'sub-{subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')

    mni = op.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_1mm_brain.nii.gz')


    # r2 com to MNI
    transformer = ants.ApplyTransforms(interpolation='NearestNeighbor')
    transformer.inputs.input_image = op.join(target_dir, f'sub-{subject}_desc-r2npc_mask.nii.gz')
    transformer.inputs.output_image = op.join(target_dir, f'sub-{subject}_space-MNI_desc-r2npc_mask.nii.gz')
    transformer.inputs.transforms = [to_mni]
    transformer.inputs.reference_image = mni
    r = transformer.run()

    # smooth r2 to MNI
    transformer = ants.ApplyTransforms(interpolation='NearestNeighbor')
    transformer.inputs.input_image = op.join(target_dir, f'sub-{subject}_desc-r2smoothnpc_mask.nii.gz')
    transformer.inputs.output_image = op.join(target_dir, f'sub-{subject}_space-MNI_desc-r2smoothnpc_mask.nii.gz')
    transformer.inputs.transforms = [to_mni]
    transformer.inputs.reference_image = mni

    r = transformer.run()

    def go_to_marius_space(im, interpolation='linear', template=None):
        
        im = image.load_img(im)
        
        if template is not None:
            im = image.resample_to_img(im, template, interpolation='nearest')
        
        new_affine = np.identity(4)
        new_affine[:3, -1] = np.linalg.inv(im.affine[:3, :3]).dot(im.affine[:3, -1])
        
        im2 = image.resample_img(im, new_affine, interpolation=interpolation)
        
        im2.affine[:3, -1] = 0.0
        
        return im2

    template = t1w = op.join(bids_folder, f'derivatives/fmriprep/sub-{subject}/anat/sub-{subject}_desc-preproc_T1w.nii.gz')

    go_to_marius_space(r2_smooth, template=t1w).to_filename(f'/data/ds-risk/zooi/for_marius/sub-{subject}_desc-smoothr2.nii.gz')

    go_to_marius_space(op.join(target_dir, f'sub-{subject}_desc-r2npc_mask.nii.gz'), template=t1w).to_filename(f'/data/ds-risk/zooi/for_marius/sub-{subject}_desc-r2peak_mask.nii.gz')

    go_to_marius_space(op.join(target_dir, f'sub-{subject}_desc-r2smoothnpc_mask.nii.gz'), template=t1w).to_filename(f'/data/ds-risk/zooi/for_marius/sub-{subject}_desc-r2smoothpeak_mask.nii.gz')

    go_to_marius_space(ips_mask, template=t1w, interpolation='nearest').to_filename(f'/data/ds-risk/zooi/for_marius/sub-{subject}_desc-npc12_mask.nii.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder,)
