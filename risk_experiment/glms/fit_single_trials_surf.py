import nibabel as nb
import pandas as pd
import numpy as np
import argparse
import os
import os.path as op
from nilearn import surface
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from itertools import product



def main(subject, session, sourcedata, space='fsnative', n_jobs=7):

    derivatives = op.join(sourcedata, 'derivatives')
    base_dir = op.join(derivatives, 'glm_stim1_surf', f'sub-{subject}',
            f'ses-{session}', 'func')
    
    if not op.exists(base_dir):
        os.makedirs(base_dir)

    runs = range(1, 9)

    behavior = []
    for run in runs:
        behavior.append(pd.read_table(op.join(
            sourcedata, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task-{run}_events.tsv')))

    behavior = pd.concat(behavior, keys=runs, names=['run'])
    behavior['subject'] = subject
    behavior = behavior.reset_index().set_index(
        ['subject', 'run', 'trial_type'])


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
    n2['modulation'] = zscore(n2['n2'])
    n2['trial_type'] = 'n_dots2'

    p2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'prob2']]
    p2 = p2[p2.prob2 == 1.0]
    p2['duration'] = 0.6
    p2['trial_type'] = 'certain2'

    events = pd.concat((stimulus1, stimulus2, n2, p2)).sort_values('onset')
    events['modulation'].fillna(1.0, inplace=True)

    # # sub-02_ses-7t2_task-task_run-1_space-fsaverage_hemi-R_bold.func

    keys = [(run, hemi) for run, hemi in product(runs, ['L', 'R'])]
    surfs = [
        op.join(sourcedata, f'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii') for run, hemi in keys]

    fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                  'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                  'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02', 
                                  'cosine03',
                                  'non_steady_state_outlier00', 'non_steady_state_outlier01', 'non_steady_state_outlier02']
    fmriprep_confounds = [
        op.join(sourcedata, f'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{run}_desc-confounds_timeseries.tsv') for run, hemi in keys]
    fmriprep_confounds = [pd.read_table(
        cf)[fmriprep_confounds_include] for cf in fmriprep_confounds]

    retroicor_confounds = [
        op.join(sourcedata, f'derivatives/physiotoolbox/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{run}_desc-retroicor_timeseries.tsv') for run, hemi in keys]
    retroicor_confounds = [pd.read_table(
        cf, header=None, usecols=range(18)) for cf in retroicor_confounds]

    confounds = [pd.concat((rcf, fcf), axis=1) for rcf, fcf in zip(retroicor_confounds, fmriprep_confounds)]
    confounds = [c.fillna(method='bfill') for c in confounds]

    t_r, n_scans = 2.3, 160
    frame_times = t_r * (np.arange(n_scans) + .5)

    betas = []

    for (run, hemi), cf, surf in zip(keys, confounds, surfs):
        print(run, hemi)
        e = events.xs(run, 0, 'run')
        Y = surface.load_surf_data(surf).T

        if len(Y) == 213:
            Y = Y[:160]
            cf = cf.iloc[:160]

        X = make_first_level_design_matrix(frame_times,
                                               events=e,
                                               hrf_model='glover',
                                               high_pass=False,
                                               drift_model=None,
                                               add_regs=cf,
                                               )


        Y = (Y / Y.mean(0) * 100)
        Y -= Y.mean(0)

        fit = run_glm(Y, X, noise_model='ols', n_jobs=n_jobs)
        r = fit[1][0.0]
        betas.append(pd.DataFrame(r.theta, index=X.columns))

    betas = pd.concat(betas, keys=keys, names=['run', 'hemi'])
    print(betas)
    betas.reset_index('run', drop=True, inplace=True)
    betas = betas.loc[(slice(None), stimulus1.trial_type), :].unstack('hemi').swaplevel(axis=1).sort_index(axis=1)
    betas = betas.loc[:, ~betas.isnull().all(0)]


    for hemi in ['L', 'R']:
        gii = nb.gifti.GiftiImage(header=nb.load(surfs[['L', 'R'].index(hemi)]).header,
                                  darrays=[nb.gifti.GiftiDataArray(row) for _, row in betas[hemi].iterrows()])

        fn_template = op.join(base_dir, 'sub-{subject}_ses-{session}_task-task_space-{space}_desc-stims1_hemi-{hemi}.pe.gii')

        gii.to_filename(fn_template.format(**locals()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--sourcedata', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, sourcedata=args.sourcedata)
