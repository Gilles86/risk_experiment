import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nb
import pandas as pd
import numpy as np
import argparse
import os
import os.path as op
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from itertools import product
from sklearn.decomposition import PCA
from scipy.stats import zscore
from risk_experiment.utils import get_confounds



def main(subject, session, bids_folder, smoothed=False, 
        pca_confounds=False,
        space='fsnative', n_jobs=14):

    derivatives = op.join(bids_folder, 'derivatives')

    runs = range(1, 9)

    ims = [
        op.join(derivatives, f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{run}_space-T1w_desc-preproc_bold.nii.gz') for run in runs]

    mask = op.join(derivatives, f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')

    base_dir = 'glm_stim1'

    if smoothed:
        base_dir += '.smoothed'

    if pca_confounds:
        base_dir += '.pca_confounds'

    base_dir = op.join(derivatives, base_dir, f'sub-{subject}',
            f'ses-{session}', 'func')
    
    if not op.exists(base_dir):
        os.makedirs(base_dir)


    behavior = []
    for run in runs:
        behavior.append(pd.read_table(op.join(
            bids_folder, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv')))

    behavior = pd.concat(behavior, keys=runs, names=['run'])
    behavior = behavior.reset_index().set_index(
        ['run', 'trial_type'])


    stimulus1 = behavior.xs('stimulus 1', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type']]
    stimulus1['duration'] = 0.6
    stimulus1['trial_type'] = stimulus1.trial_nr.map(lambda trial: f'trial_{trial}')

    print(stimulus1)
    
    stimulus2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type']]
    stimulus2['duration'] = 0.6

    n2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'n2']]
    n2['duration'] = 0.6
    def zscore(n):
        return (n - n.mean()) / n.std()
    # n2['modulation'] = zscore(n2['n2'])
    n2['modulation'] = n2['n2'].groupby(['run']).apply(zscore)
    n2['trial_type'] = 'n_dots2'

    p2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'prob2']]
    p2 = p2[p2.prob2 == 1.0]
    p2['duration'] = 0.6
    p2['trial_type'] = 'certain2'

    events = pd.concat((stimulus1, stimulus2, n2, p2)).sort_index()
    events['modulation'].fillna(1.0, inplace=True)
    print(events.loc[1].sort_values('onset').head(100))

    confounds = get_confounds(subject, session, bids_folder, None,
    pca=pca_confounds)

    t_r, n_scans = 2.3, 160
    frame_times = t_r * (np.arange(n_scans) + .5)

    if smoothed:
        smoothing_fwhm = 5.0
    else:
        smoothing_fwhm = 0.0

    model = FirstLevelModel(t_r=2.3, slice_time_ref=.5, signal_scaling=False, drift_model=None, 
            mask_img=mask,
                        smoothing_fwhm=smoothing_fwhm)

    single_trial_betas = []

    for run in runs:
        im = image.math_img('(im / im.mean(-1)[..., np.newaxis]) * 100 - 100', im=ims[run-1])
        model.fit(im, events.loc[run], confounds[run-1])


        for trial in range(1+(run-1)*24, 1+run*24):
            print(trial)
            single_trial_betas.append(model.compute_contrast(f'trial_{trial}', output_type='effect_size'))


    single_trial_betas = image.concat_imgs(single_trial_betas)
    single_trial_betas.to_filename(op.join(base_dir, f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims1_pe.nii.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pca_confounds', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder, smoothed=args.smoothed,
            pca_confounds=args.pca_confounds)
