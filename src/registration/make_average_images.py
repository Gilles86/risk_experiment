import os
import os.path as op
import argparse
import glob
from nilearn import image
import numpy as np
from tqdm import tqdm

def main(modality, field_strength, bids_folder):

    source = op.join(bids_folder, 'derivatives', 'registration')

    template = op.join(source, 'sub-*', f'ses-{field_strength}*',
            'anat', f'sub-*_ses-*_space-MNI152NLin2009cAsym_desc-registered_{modality}.nii.gz')

    ims = [image.load_img(im) for im in glob.glob(template)]

    for im in ims:
        print(im.get_filename())


    n_voxels = [np.prod(im.shape) for im in ims]
    largest_fov = np.argmax(n_voxels)

    from collections import deque
    ims = deque(ims)
    ims.rotate(-largest_fov)

    mean_img = ims[0]

    for ix in tqdm(range(1, len(ims))):
        mean_img = image.math_img('mean_img + im',
                mean_img=mean_img,
                im=image.resample_to_img(ims[ix], mean_img))
        ims[ix].uncache()

    mean_img = image.math_img(f'mean_img / {len(ims)}',
            mean_img=mean_img)

    target_fn = op.join(source, 'group')

    if not op.exists(target_fn):
        os.makedirs(target_fn)

    target_fn = op.join(target_fn, f'group_space-MNI152NLin2009cAsym_field-{field_strength}_{modality}.nii.gz')

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
