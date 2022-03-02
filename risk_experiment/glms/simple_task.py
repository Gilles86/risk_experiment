import pandas as pd
import numpy as np
import argparse
import os
import os.path as op
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix


def main(subject, session, sourcedata):
    runs = range(1, 9)
    behavior = []

    for run in runs:
        behavior.append(pd.read_table(op.join(
            sourcedata, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task-{run}_events.tsv')))

    behavior = pd.concat(behavior, keys=runs, names=['run'])
    behavior['subject'] = subject
    behavior = behavior.reset_index().set_index(
        ['subject', 'run', 'trial_type'])

    ims = [
        op.join(sourcedata, f'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{run}_space-T1w_desc-preproc_bold.nii.gz') for run in runs]

    mask_img = op.join(sourcedata, f'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')

    fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                  'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                  'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02', 
                                  'cosine03',
                                  'non_steady_state_outlier00', 'non_steady_state_outlier01', 'non_steady_state_outlier02']
    fmriprep_confounds = [
        op.join(sourcedata, f'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{run}_desc-confounds_timeseries.tsv') for run in runs]
    fmriprep_confounds = [pd.read_table(
        cf)[fmriprep_confounds_include] for cf in fmriprep_confounds]

    retroicor_confounds = [
        op.join(sourcedata, f'derivatives/physiotoolbox/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-{run}_desc-retroicor_timeseries.tsv') for run in runs]
    retroicor_confounds = [pd.read_table(
        cf, header=None, usecols=range(18)) for cf in retroicor_confounds]

    confounds = [pd.concat((rcf, fcf), axis=1) for rcf, fcf in zip(retroicor_confounds, fmriprep_confounds)]
    confounds = [c.fillna(method='bfill') for c in confounds]

    print(behavior)

    print(confounds)
    model = FirstLevelModel(t_r=2.3, slice_time_ref=.5, signal_scaling=False, drift_model=None, 
            mask_img=mask_img,
                        smoothing_fwhm=0.0)

    stimulus1 = behavior.xs('stimulus 1', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type']]
    stimulus1['duration'] = 0.6
    stimulus2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type']]
    stimulus2['duration'] = 0.6

    n1 = behavior.xs('stimulus 1', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'n1']]
    n1['duration'] = 0.6
    def zscore(n):
        return (n - n.mean()) / n.std()
    n1['modulation'] = zscore(n1['n1'])
    n1['trial_type'] = 'n_dots1'

    n2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'n2']]
    n2['duration'] = 0.6
    def zscore(n):
        return (n - n.mean()) / n.std()
    n2['modulation'] = zscore(n2['n2'])
    n2['trial_type'] = 'n_dots2'

    p1 = behavior.xs('stimulus 1', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'prob1']]
    p1 = p1[p1.prob1 == 1.0]
    p1['duration'] = 0.6
    p1['trial_type'] = 'certain1'

    p2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'prob2']]
    p2 = p2[p2.prob2 == 1.0]
    p2['duration'] = 0.6
    p2['trial_type'] = 'certain2'

    events = pd.concat((stimulus1, stimulus2, n1, n2, p1, p2)).sort_values('onset')
    events['modulation'].fillna(1.0, inplace=True)
    print(events)


    model.fit(ims, [r for _, r in events.groupby(['run'])], confounds)

    zmaps = {}

    results_dir = op.join(sourcedata, 'derivatives', 'glm', 'simple_task', f'sub-{subject}', f'ses-{session}')

    if not op.exists(results_dir):
        os.makedirs(results_dir)

    for key in ['stimulus 1', 'stimulus 2', 'n_dots1', 'n_dots2', 'certain1', 'certain2']:
        zmaps[key] = model.compute_contrast(key)
    
        zmaps[key].to_filename(op.join(results_dir, f'sub-{subject}_ses-{session}_zmap_{key}.nii.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--sourcedata', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, sourcedata=args.sourcedata)
