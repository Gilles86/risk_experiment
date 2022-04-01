import os
import os.path as op
import argparse
import glob
from nilearn import image
import numpy as np
from tqdm import tqdm

def main(modality, field_strength, bids_folder):


    if modality == 'T1w':
        source = op.join(bids_folder, 'derivatives', 'fmriprep')
        template = op.join(source, 'sub-*',
                'anat', f'sub-*_space-MNI152NLin2009cAsym_desc-preproc_{modality}.nii.gz')

        target_fn = op.join(bids_folder, 'derivatives', 'registration', 'group')

    else:
        source = op.join(bids_folder, 'derivatives', 'registration')
        template = op.join(source, 'sub-*', f'ses-{field_strength}*',
                'anat', f'sub-*_ses-*_space-MNI152NLin2009cAsym_desc-registered_{modality}.nii.gz')
        target_fn = op.join(source, 'group')

    print(sorted(glob.glob(template)))

    ims = [image.load_img(im) for im in glob.glob(template)]

    for im in ims:
        print(im.get_filename())


    n_voxels = [np.prod(im.shape) for im in ims]
    largest_fov = np.argmax(n_voxels)

    from collections import deque
    ims = deque(ims)
    ims.rotate(-largest_fov)

    ims = [image.math_img('im / np.mean(im[im!=0])', im=im) for im in ims]

    mean_img = ims[0]
    sum_img = image.new_img_like(ims[0], np.zeros(ims[0].shape) )

    for ix in tqdm(range(1, len(ims))):
        im = image.resample_to_img(ims[ix], mean_img)
        mean_img = image.math_img('mean_img + im',
                mean_img=mean_img,
                im=im)
        sum_img = image.math_img('sum_img + (im != 0)', sum_img=sum_img, im=im)
        ims[ix].uncache()

    mean_img = image.math_img(f'np.nan_to_num(np.clip(mean_img / sum_img, 0, 10))',
            mean_img=mean_img, sum_img=sum_img)


    if not op.exists(target_fn):
        os.makedirs(target_fn)

    target_fn = op.join(target_fn, f'group_space-MNI152NLin2009cAsym_field-{field_strength}_{modality}.nii.gz')
    print(target_fn)

    mean_img.to_filename(target_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modality', default=None)
    parser.add_argument('field_strength', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()
    print(args)

    main(args.modality, args.field_strength, args.bids_folder)
