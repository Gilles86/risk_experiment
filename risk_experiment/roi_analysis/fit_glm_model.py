import os
import os.path as op
from risk_experiment.utils.data import Subject, get_all_subjects
from tqdm import tqdm
import pandas as pd
from scipy.stats import zscore
import nideconv
import argparse

def main(model_label, roi, session, bids_folder):

    target_dir = op.join(bids_folder, 'derivatives', 'roi_analysis', f'model-{model_label}', roi)

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    ts, events, behavior, confounds  = get_data(model_label, roi, bids_folder, session)

    if model_label.startswith('n1_n2'):
        n1 = events[events.event == 'n1'].join(behavior[['n1', 'ev1']]).rename(columns={'n1':'n', 'ev1':'ev'})[['onset', 'n', 'ev']]
        n1['event_type'] = 'n1'
        n2 = events[events.event == 'n2'][['onset']].join(behavior[['n2', 'ev2', 'ev_diff']]).rename(columns={'n2':'n', 'ev2':'ev'})[['onset', 'n', 'ev', 'ev_diff']]
        n2['event_type'] = 'n2'

        for key in ['n', 'ev']:
            n1[key] = n1.groupby(['subject'], group_keys=False)[key].apply(zscore)

        for key in ['n', 'ev', 'ev_diff']:
            n2[key] = n2.groupby(['subject'], group_keys=False)[key].apply(zscore)

        onsets = pd.concat((n1, n2))
    elif model_label.startswith('neural'):

        n1 = events[events.event == 'n1'].join(behavior[['n1', 'sd']]).rename(columns={'n1':'n'})[['onset', 'n', 'sd']]
        n1['event_type'] = 'n1'
        n2 = events[events.event == 'n2'][['onset',]].join(behavior[['n2', 'sd']]).rename(columns={'n2':'n'})[['onset', 'n', 'sd']]
        n2['event_type'] = 'n2'

        for key in ['n', 'sd']:
            n1[key] = n1.groupby(['subject'], group_keys=False)[key].apply(zscore)
            n2[key] = n2.groupby(['subject'], group_keys=False)[key].apply(zscore)

        onsets = pd.concat((n1, n2))
    elif model_label.startswith('pupil'):
        n1 = events[events.event == 'n1'].join(behavior[['n1', 'pupil']]).rename(columns={'n1':'n'})[['onset', 'n', 'pupil']]
        n1['event_type'] = 'n1'
        n2 = events[events.event == 'n2'][['onset']].join(behavior[['n2', 'pupil']]).rename(columns={'n2':'n'})[['onset', 'n', 'pupil']]
        n2['event_type'] = 'n2'

        onsets = pd.concat((n1, n2))

    else:
        raise NotImplementedError

    grf = nideconv.GroupResponseFitter(ts, onsets=onsets, input_sample_rate=1./2.3, confounds=confounds, concatenate_runs=False)
    
    if model_label == 'n1_n2':
        grf.add_event('n1', interval=[-4.0, 19], basis_set='dct', n_regressors=8)
        grf.add_event('n2', interval=[-4.0, 19], basis_set='dct', n_regressors=8)
    elif model_label == 'n1_n2_n':
        grf.add_event('n1', interval=[-4.0, 19], basis_set='dct', n_regressors=8, covariates='n')
        grf.add_event('n2', interval=[-4.0, 19], basis_set='dct', n_regressors=8, covariates='n')
    elif model_label == 'n1_n2_ev':
        grf.add_event('n1', interval=[0.0, 18.4], basis_set='dct', n_regressors=7, covariates='ev')
        grf.add_event('n2', interval=[0.0, 18.4], basis_set='dct', n_regressors=7, covariates='ev')
    elif model_label == 'n1_n2_evdiff':
        grf.add_event('n1', interval=[0.0, 18.4], basis_set='dct', n_regressors=7, covariates='ev')
        grf.add_event('n2', interval=[0.0, 18.4], basis_set='dct', n_regressors=7, covariates='ev_diff')
    elif model_label == 'neural1':
        grf.add_event('n1', interval=[-6., 19], basis_set='dct', n_regressors=8, covariates=['n', 'sd'])
        grf.add_event('n2', interval=[-6., 19], basis_set='dct', n_regressors=8, covariates='n')
    elif model_label == 'neural2':
        grf.add_event('n1', interval=[-6., 19], basis_set='dct', n_regressors=8, covariates=['n', 'sd'])
        grf.add_event('n2', interval=[-6., 19], basis_set='dct', n_regressors=8, covariates=['n', 'sd'])
    elif model_label == 'neural3':
        grf.add_event('n1', interval=[-6., 19], basis_set='dct', n_regressors=8, covariates='sd')
        grf.add_event('n2', interval=[-6., 19], basis_set='dct', n_regressors=8)
    elif model_label == 'pupil_baseline':
        grf.add_event('n1', interval=[-6., 19], basis_set='dct', n_regressors=8, covariates='pupil')
        grf.add_event('n2', interval=[-6., 19], basis_set='dct', n_regressors=8)

    grf.fit()
    grf.plot_groupwise_timecourses()

    fac = grf.plot_groupwise_timecourses()[0]

    fac.savefig(op.join(target_dir, f'ses-{session}_plot_all.pdf'))
    fac.savefig(op.join(target_dir, f'ses-{session}_plot_all.png'))

    for event in grf.events:
        fac = grf.plot_groupwise_timecourses(event)[0]
        fac.savefig(op.join(target_dir, f'ses-{session}_plot_{event}.pdf'))
        fac.savefig(op.join(target_dir, f'ses-{session}_plot_{event}.png'))


    if model_label == 'uncertainty_median_split':
        fac = grf.plot_groupwise_timecourses(['n1 (uncertain)', 'n1 (certain)'])[0]
        fac.savefig(op.join(target_dir, f'ses-{session}_plot_certain_uncertain.pdf'))
        fac.savefig(op.join(target_dir, f'ses-{session}_plot_certain_uncertain.png'))

    tcs = grf.get_subjectwise_timecourses()

    tcs.to_csv(op.join(target_dir, f'ses-{session}_timeseries.tsv'), sep='\t')



def get_data(model_label, roi, bids_folder, session='7t2', pca=False):
    subjects = get_all_subjects(bids_folder)

    ts = pd.concat([sub.get_roi_timeseries(session, roi, pca=pca) for sub in subjects]).droplevel('session').droplevel('frame')
    events = pd.concat([sub.get_fmri_events(session) for sub in subjects], keys=[sub.subject for sub in subjects], names=['subject']).set_index('trial_nr', append=True).reset_index('session', drop=True)
    behavior = pd.concat([sub.get_behavior(sessions=session) for sub in subjects]).reset_index('session', drop=True)

    confounds = pd.concat([pd.concat(sub.get_confounds(session, pca=False), keys=range(1, 9), names=['run']) for sub in subjects], keys=[sub.subject for sub in subjects], names=['subject'])

    if model_label.startswith('neural'):
        decoding_info = pd.concat([sub.get_decoding_info(session, mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
        behavior = behavior.join(decoding_info)
        behavior['sd'] = behavior.groupby('subject', group_keys=False)['sd'].apply(zscore)
        behavior['median_split(sd)'] = behavior.groupby(['subject', 'session'], group_keys=False)['sd'].apply(lambda d: d>d.quantile()).map({True:'High neural uncertainty', False:'Low neural uncertainty'})
    elif model_label.startswith('pupil_baseline'):
        pupil = pd.read_csv(op.join(bids_folder, 'derivatives', 'pupil', 'model-n1_n2_n', 'pupil_pre_post12.tsv'), sep='\t', index_col=[0,1,2,3], dtype={'subject':str})
        pupil_baseline = pupil.xs('n1', 0, 'event type').xs('pre', 0, 'prepost')
        pupil_baseline['pupil'] = pupil_baseline.groupby('subject', group_keys=False)['pupil'].apply(zscore)
        behavior = behavior.join(pupil_baseline)

    return ts, events, behavior, confounds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('roi', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    parser.add_argument('--session', default='7t2')
    args = parser.parse_args()

    main(args.model_label, args.roi, args.session, args.bids_folder)