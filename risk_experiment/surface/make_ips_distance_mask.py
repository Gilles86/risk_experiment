from nibabel import gifti
from tqdm import tqdm
import argparse
import os.path as op
import nibabel as nb
from nilearn import surface
import numpy as np
import cortex


def main(subject, bids_folder):

    (pts_l, poly_l), (pts_r, poly_r) = cortex.db.get_surf(f'sub-{subject}', 'pia')
    surf_l = cortex.polyutils.Surface(pts_l, poly_l)
    surf_r = cortex.polyutils.Surface(pts_r, poly_r)


    for hemi, surf in [('lh', surf_l), ('rh', surf_r)]:
        im = nb.load(op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subject}', 'surf', f'{hemi}.wang15_ips.mgz'))
        mask = np.squeeze(im.get_data().astype(bool))
        print(mask.sum())

        ss = surf.create_subsurface(mask)
        nlverts = len(ss.pts)

        dist_matrix = np.zeros((nlverts, nlverts))

        for i in tqdm(range(len(dist_matrix))):
            dist_matrix[i] = ss.geodesic_distance(i)

        v1 = dist_matrix[np.triu_indices(nlverts, k = 1)]
        v2 = dist_matrix[np.tril_indices(nlverts, k = -1)]
        v = (v1 + v2) / 2.

        # see https://stackoverflow.com/questions/17527693/transform-the-upper-lower-triangular-part-of-a-symmetric-matrix-2d-array-into/58806626#58806626
        dist_matrix_ = np.zeros((nlverts,nlverts))
        dist_matrix_[np.triu_indices(nlverts, k = 1)] = v
        dist_matrix_ = dist_matrix_ + dist_matrix_.T

        print(dist_matrix - dist_matrix_)
        print(np.array_equal(dist_matrix, dist_matrix_))

        # im = gifti.GiftiImage(darrays=[gifti.GiftiDataArray(v)])
        # im.to_filename(op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subject}', 'surf', f'{hemi}.wang15_ips_distance.gii'))

        new_im = nb.MGHImage(v, im.affine, im.header)
        new_im.to_filename(op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subject}', 'surf', f'{hemi}.wang15_ips_distance.mgz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.bids_folder, )
