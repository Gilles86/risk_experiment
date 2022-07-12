import os
import pandas as pd
import os.path as op
import argparse
import cortex
import os.path as op
from nilearn import surface
import matplotlib.pyplot as plt
import numpy as np
from risk_experiment.utils.data import get_all_subjects
from nibabel import gifti
from risk_experiment.utils.surface import transform_data


def main(bids_folder='/data/ds-risk'):

    target_dir = op.join(bids_folder, 'derivatives', 'npc_com')
    if not op.exists(target_dir):
        os.makedirs(target_dir)
    
    subjects = get_all_subjects()

    df = []
    for subject in subjects:
        coords = get_npcr_coordinate(subject, bids_folder=bids_folder)

        df.append({'subject':subject, 'x':coords[0], 'y':coords[1]})

        surf = cortex.Surface(*cortex.db.get_surf('fsaverage', 'flat', merge=False, hemisphere='right'))

        ball = surf.get_euclidean_ball(coords, radius=5).astype(float)

        darrays = [gifti.GiftiDataArray(ball)]
        image = gifti.GiftiImage(darrays=darrays)  

        fn = op.join(target_dir, f'sub-{subject}_space-fsaverage_desc-npcr_hemi-R_com.gii')
        image.to_filename(fn)

        fsnative_fn = op.join(target_dir, f'sub-{subject}_space-fsnative-npcr_hemi-R_com.gii')
        transform_data(fn, 'fsaverage', bids_folder, target_fn=fsnative_fn, target_subject=f'sub-{subject}')

        print(ball)
    
    df = pd.DataFrame(df)


    df.set_index('subject').to_csv(op.join(target_dir, 'coms.tsv'), sep='\t')

    


def get_npcr_coordinate(subject, session='3t2', hemi='r', space='fsaverage',
        bids_folder='/data/ds-risk'):
    
    if (hemi != 'r') | (space != 'fsaverage'):
        raise NotImplementedError()

    dir_ = op.join(bids_folder, 'derivatives', 'encoding_model', f'sub-{subject}', f'ses-{session}',
               'func')
    
    r2 = op.join(dir_, f'sub-{subject}_ses-{session}_desc-r2.volume.optim_space-fsaverage_hemi-R.func.gii')
    r2 = surface.load_surf_data(r2)
    thr = np.nanpercentile(r2, 80)
    
    npc1_mask = surface.load_surf_data(op.join(bids_folder, 'derivatives', 
                                              'surface_masks.v3', 'desc-NPC1_R_space-fsaverage_hemi-rh.label.gii')).astype(bool)
    
    npc2_mask = surface.load_surf_data(op.join(bids_folder, 'derivatives', 
                                              'surface_masks.v3', 'desc-NPC2_R_space-fsaverage_hemi-rh.label.gii')).astype(bool)

    npc_mask = (npc1_mask | npc2_mask)

    r2_ = r2[npc_mask].copy()
    r2_[r2_<thr] = 0.0
    
    right = cortex.db.get_surf(f'fsaverage', 'flat', merge=False, hemisphere='right')
    
    right = cortex.Surface(*right)
    
    coord = (r2_[:, np.newaxis]*right.pts[npc_mask]).sum(0) / r2_.sum()
    
    return coord

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(bids_folder=args.bids_folder, )
