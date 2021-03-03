import numpy as np
from nilearn import plotting, image
import pandas as pd
import argparse
import os.path as op
import os
import matplotlib.pyplot as plt

def main(subject, session, bids_folder='/data2/ds-risk'):

    derivatives = op.join(bids_folder, 'derivatives')

    #Make columns
    columns = []
    for n, modality in zip([3, 4, 2], ['cardiac', 'respiratory', 'interaction']):    
        for order in range(1, n+1):
            columns += [(modality, order, 'sin'), (modality, order, 'cos')]

    columns = pd.MultiIndex.from_tuples(columns, names=['modality', 'order', 'type'])


    # Prepare T1w
    t1w = op.join(derivatives, 'fmriprep', f'sub-{subject}', 'anat', f'sub-{subject}_desc-preproc_T1w.nii.gz')

    if not op.exists(t1w):
        print(f'{t1w} does not exist!')
        t1w = op.join(derivatives, 'fmriprep', f'sub-{subject}', 'ses-7t1', 'anat', f'sub-{subject}_ses-7t1_desc-preproc_T1w.nii.gz')
            
    t1w_mask = op.join(derivatives, 'fmriprep', f'sub-{subject}', 'anat', f'sub-{subject}_desc-brain_mask.nii.gz')
    if not op.exists(t1w_mask):
        t1w_mask = op.join(derivatives, 'fmriprep', f'sub-{subject}', 'ses-7t1', 'anat', f'sub-{subject}_ses-7t1_desc-brain_mask.nii.gz')

    t1w = image.math_img('t1w*mask', t1w=t1w, mask=t1w_mask)
    t1w = image.math_img('np.clip(t1w, 0, np.percentile(t1w, 95))', t1w=t1w)

    # Get session info
    if session[-1] == '1':
        runs = range(1, 5)
        task = 'mapper'
    elif session[-1] == '2':
        runs = range(1, 9)
        task = 'task'

    # Make figure folder    
    figure_dir = op.join(derivatives, 'physioplots', f'sub-{subject}', f'ses-{session}', 'func')
    if not op.exists(figure_dir):
        os.makedirs(figure_dir)
        
    # Loop over runs
    for run in runs:
        confounds = pd.read_csv(op.join(derivatives, 'physiotoolbox', f'sub-{subject}', f'ses-{session}', 'func', 
                                 f'sub-{subject}_ses-{session}_task-{task}_run-{run}_desc-retroicor_timeseries.tsv'),
                               usecols=range(18), names=columns, sep='\t')
        
        im = image.load_img(op.join(derivatives, 'fmriprep', f'sub-{subject}', f'ses-{session}', 'func', 
                                 f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-T1w_desc-preproc_bold.nii.gz'))
        
        # if session[0] == '3':
            # im = image.smooth_img(im, 5)

        im = image.clean_img(im,  detrend=False, standardize=True)
        clean_im_all = image.clean_img(im,  confounds=confounds.values, detrend=False, standardize=False)
        clean_im_resp = image.clean_img(im,  confounds=confounds['respiratory'].values, detrend=False, standardize=False)
        clean_im_cardiac = image.clean_img(im,  confounds=confounds['cardiac'].values, detrend=False, standardize=False)
        clean_im_interaction = image.clean_img(im,  confounds=confounds['interaction'].values, detrend=False, standardize=False)
        
        r2_all = image.math_img('1 - (np.var(clean_im, -1) / np.var(im, -1))', im=im, clean_im=clean_im_all)
        r2_resp = image.math_img('1 - (np.var(clean_im, -1) / np.var(im, -1))', im=im, clean_im=clean_im_resp)
        r2_cardiac = image.math_img('1 - (np.var(clean_im, -1) / np.var(im, -1))', im=im, clean_im=clean_im_cardiac)
        r2_interaction = image.math_img('1 - (np.var(clean_im, -1) / np.var(im, -1))', im=im, clean_im=clean_im_interaction)
        
        r2_all.to_filename(op.join(derivatives, 'physiotoolbox', f'sub-{subject}', f'ses-{session}', 'func', 
                                  f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-T1w_desc-r2all_bold.nii.gz'))
        r2_resp.to_filename(op.join(derivatives, 'physiotoolbox', f'sub-{subject}', f'ses-{session}', 'func', 
                                  f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-T1w_desc-r2resp_bold.nii.gz'))
        r2_cardiac.to_filename(op.join(derivatives, 'physiotoolbox', f'sub-{subject}', f'ses-{session}', 'func', 
                                  f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-T1w_desc-r2cardiac_bold.nii.gz'))
        
        r2_interaction.to_filename(op.join(derivatives, 'physiotoolbox', f'sub-{subject}', f'ses-{session}', 'func', 
                                  f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-T1w_desc-r2interaction_bold.nii.gz'))    
        
        
        n_slices = r2_all.shape[2]
        slices = 8

        plotting.plot_stat_map(r2_cardiac, t1w, display_mode='z', threshold=0.2, figure=run, axes=(0, .66, 1, .33),
                vmax=.8,
                              cut_coords=slices)

        plotting.plot_stat_map(r2_resp, t1w, display_mode='z', threshold=0.2, cmap='viridis', figure=run, 
                              axes=(0., 0.33, 1, .33),
                              vmax=.8,
                            cut_coords=slices)

        plotting.plot_stat_map(r2_interaction, t1w, display_mode='z', threshold=0.125, cmap='Blues', figure=run, 
                              axes=(0., 0.0, 1, .33),
                              vmax=.8,
                               cut_coords=slices)

        plt.gcf().set_size_inches((12, 6))

        # plt.tight_layout()

        plt.savefig(op.join(derivatives, 'physioplots', f'sub-{subject}', f'ses-{session}', 'func', 
                                  f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-T1w_desc-r2.png'), 
                   )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--sourcedata', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, sourcedata=args.sourcedata)
