import argparse
import nibabel as nb
import scipy.stats as ss
import numpy as np
import os.path as op
from nilearn import surface
import cortex
from nipype.interfaces.freesurfer import SurfaceTransform


def main(subject, bids_folder='/data'):

    subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')

    masks = [op.join(subjects_dir, f'sub-{subject}', 'surf', f'{hemi}.npc.mgz') for hemi in ['lh', 'rh']]

    masks = [surface.load_surf_data(mask).astype(np.bool) for mask in masks]

    surfs = [cortex.polyutils.Surface(*d)
             for d in cortex.db.get_surf(f'sub-{subject}', "fiducial")]

    def get_euclidean_center_vertex(ss):

        dist = np.sqrt(((ss.pts - ss.pts.mean(0))**2).sum(1))

        center_vertex = np.argmin(dist)

        return center_vertex

    def get_euclidean_distance_matrix(ss, center_vertex):

        return np.sqrt(((ss.pts - ss.pts[center_vertex])**2).sum(1))


    subsurfs = [ss.create_subsurface(mask) for ss, mask in zip(surfs, masks)]
    center_vertices = [get_euclidean_center_vertex(ss) for ss in subsurfs]

    geod_distance_matrices = [ss.geodesic_distance(cv) for ss, cv in zip(subsurfs, center_vertices)]
    geod_distance_matrices = [ss.lift_subsurface_data(dm) for ss, dm in zip(subsurfs, geod_distance_matrices)]

    eucl_distance_matrices = [get_euclidean_distance_matrix(ss, cv) for ss, cv in zip(subsurfs, center_vertices)]
    eucl_distance_matrices = [ss.lift_subsurface_data(dm) for ss, dm in zip(subsurfs, eucl_distance_matrices)]

    print([ss.pearsonr(gdm[gdm!=0.0], edm[edm!=0.0]) for gdm, edm in zip(geod_distance_matrices, eucl_distance_matrices)])

    def transform_surface(in_file,
            out_file, 
            target_subject,
            hemi,
            source_subject='fsaverage'):

        print(subjects_dir)
        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = out_file
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = target_subject
        sxfm.inputs.hemi = hemi

        r = sxfm.run()
        return r


    for gdm, edm, hemi in zip(geod_distance_matrices, eucl_distance_matrices, ['lh', 'rh']):

        for dm, label in zip([gdm, edm], ['geodesic', 'euclidean']):
            im = nb.MGHImage(dm.astype(np.float32), np.identity(4))
            fn = op.join(subjects_dir, f'sub-{subject}', 'surf', f'{hemi}.npc_{label}distance.mgz')
            im.to_filename(fn)

            fsaverage_fn = op.join(subjects_dir, f'sub-{subject}', 'surf', f'{hemi}.npc_{label}distance_space-fsaverage.mgz') 

            transform_surface(fn, fsaverage_fn, 'fsaverage', 
                    hemi, f'sub-{subject}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.bids_folder, )
