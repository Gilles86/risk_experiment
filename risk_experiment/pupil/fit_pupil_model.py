import argparse
import nideconv
from risk_experiment.utils.data import Subject, get_all_subjects, get_all_behavior, get_all_subject_ids, get_all_pupil_data
from risk_experiment.utils.math import resample_run

import pandas as pd
from scipy.stats import zscore
import os
import os.path as op
from tqdm import tqdm

def main(model_label, bids_folder):

    target_dir = op.join(bids_folder, 'derivatives', 'pupil', f'model-{model_label}')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    pupil, saccades,  blinks, behavior, events = get_data(model_label, bids_folder)

    onsets = pd.concat((blinks, saccades), keys=['blink', 'saccade'], names=['event_type']).reset_index('event_type')[['event_type', 'onset']]

    # events = events.loc[events.event == 'n1', ['onset']].join(behavior[['median_split_uncertainty', 'z_uncertainty']])

    if model_label.startswith('n1_n2'):
        n1 = events[events.event == 'n1'].join(behavior[['n1']]).rename(columns={'n1':'n'})[['onset', 'n']]
        n1['event_type'] = 'n1'
        n2 = events[events.event == 'n2'][['onset']].join(behavior[['n2']]).rename(columns={'n2':'n'})[['onset', 'n']]
        n2['event_type'] = 'n2'

        onsets = pd.concat((onsets, n1, n2))
    if model_label == 'uncertainty':
        n1 = events[events.event == 'n1'].join(behavior[['n1', 'z_uncertainty']]).rename(columns={'n1':'n'})[['onset', 'n', 'z_uncertainty']]
        n1['event_type'] = 'n1'
        n2 = events[events.event == 'n2'][['onset']].join(behavior[['n2', 'z_uncertainty']]).rename(columns={'n2':'n'})[['onset', 'n', 'z_uncertainty']]
        n2['event_type'] = 'n2'

        onsets = pd.concat((onsets, n1, n2))

    if model_label == 'uncertainty_median_split':
        n1 = events[events.event == 'n1'].join(behavior[['n1', 'median_split_uncertainty']]).rename(columns={'n1':'n'})[['onset', 'n', 'median_split_uncertainty']]
        n1_certain = n1[n1['median_split_uncertainty'] == 'low uncertainty']
        n1_uncertain = n1[n1['median_split_uncertainty'] == 'high uncertainty']
        n1_certain['event_type'] = 'n1 (certain)'
        n1_uncertain = n1[n1['median_split_uncertainty'] == 'high uncertainty']
        n1_uncertain['event_type'] = 'n1 (uncertain)'
        n2 = events[events.event == 'n2'][['onset']].join(behavior[['n2']]).rename(columns={'n2':'n'})[['onset', 'n']]
        n2['event_type'] = 'n2'

        onsets = pd.concat((onsets, n1_certain, n1_uncertain, n2))

    if model_label.startswith('neural'):

        n1 = events[events.event == 'n1'].join(behavior[['n1', 'sd']]).rename(columns={'n1':'n'})[['onset', 'n', 'sd']]
        n1['event_type'] = 'n1'
        n2 = events[events.event == 'n2'][['onset',]].join(behavior[['n2', 'sd']]).rename(columns={'n2':'n'})[['onset', 'n', 'sd']]
        n2['event_type'] = 'n2'

        onsets = pd.concat((onsets, n1, n2))

    grf = nideconv.GroupResponseFitter(pupil['pupil'], onsets=onsets, input_sample_rate=10)
    grf.add_event('saccade', interval=[-0.5, 4.5], basis_set='dct', n_regressors=12)
    grf.add_event('blink', interval=[-0.5, 4.5], basis_set='dct', n_regressors=12)

    if model_label == 'n1_n2':
        grf.add_event('n1', interval=[-3.5, 6.5], basis_set='dct', n_regressors=12)
        grf.add_event('n2', interval=[-3.5, 6.5], basis_set='dct', n_regressors=12)
    elif model_label == 'n1_n2_n':
        grf.add_event('n1', interval=[-3.5, 6.5], basis_set='dct', n_regressors=12, covariates='n')
        grf.add_event('n2', interval=[-3.5, 6.5], basis_set='dct', n_regressors=12, covariates='n')
    elif model_label == 'uncertainty':
        grf.add_event('n1', interval=[-3., 6.5], basis_set='dct', n_regressors=20, covariates=['n', 'z_uncertainty'])
        grf.add_event('n2', interval=[-3., 6.5], basis_set='dct', n_regressors=20, covariates=['n', 'z_uncertainty'])
    elif model_label == 'uncertainty_median_split':
        grf.add_event('n1 (uncertain)', interval=[-3., 6.5], basis_set='dct', n_regressors=8, covariates=['n'])
        grf.add_event('n1 (certain)', interval=[-3., 6.5], basis_set='dct', n_regressors=8, covariates=['n'])
        grf.add_event('n2', interval=[-3., 6.5], basis_set='dct', n_regressors=8, covariates='n')
    elif model_label == 'neural1':
        grf.add_event('n1', interval=[-3., 6.5], basis_set='dct', n_regressors=20, covariates=['n', 'sd'])
        grf.add_event('n2', interval=[-3., 6.5], basis_set='dct', n_regressors=20, covariates='n')
    elif model_label == 'neural2':
        grf.add_event('n1', interval=[-3., 6.5], basis_set='dct', n_regressors=20, covariates=['n', 'sd'])
        grf.add_event('n2', interval=[-3., 6.5], basis_set='dct', n_regressors=20, covariates=['n', 'sd'])


    grf.fit()

    fac = grf.plot_groupwise_timecourses()[0]

    fac.savefig(op.join(target_dir, 'plot_all.pdf'))
    fac.savefig(op.join(target_dir, 'plot_all.png'))

    for event in grf.events:
        fac = grf.plot_groupwise_timecourses(event)[0]
        fac.savefig(op.join(target_dir, f'plot_{event}.pdf'))
        fac.savefig(op.join(target_dir, f'plot_{event}.png'))


    if model_label == 'uncertainty_median_split':
        fac = grf.plot_groupwise_timecourses(['n1 (uncertain)', 'n1 (certain)'])[0]
        fac.savefig(op.join(target_dir, f'plot_certain_uncertain.pdf'))
        fac.savefig(op.join(target_dir, f'plot_certain_uncertain.png'))

    tcs = grf.get_subjectwise_timecourses()

    tcs.to_csv(op.join(target_dir, 'timeseries.tsv'), sep='\t')


def get_data(model_label, bids_folder):
    subjects = get_all_subjects(bids_folder)

    pupil = []
    saccades = []
    blinks = []

    for sub in tqdm(subjects):
        try:
            # pupil.append(sub.get_pupil('3t2'))
            saccades.append(sub.get_saccades('3t2'))
            blinks.append(sub.get_blinks('3t2'))
        except Exception as e:
            print(f'issue with subject {sub.subject}: {e}')

    # pupil = pd.concat(pupil)
    pupil = get_all_pupil_data(bids_folder)
    saccades = pd.concat(saccades)
    blinks = pd.concat(blinks)       
    events = pd.concat([sub.get_fmri_events('3t2') for sub in subjects], keys=[sub.subject for sub in subjects], names=['subject']).set_index('trial_nr', append=True).reset_index('session', drop=True)


    behavior = pd.concat([sub.get_behavior(sessions='3t2') for sub in subjects]).reset_index('session', drop=True)

    if model_label.startswith('neural'):
        decoding_info = pd.concat([sub.get_decoding_info('3t2', mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
        behavior = behavior.join(decoding_info)
        behavior['sd'] = behavior.groupby('subject', group_keys=False)['sd'].apply(zscore)
        behavior['median_split(sd)'] = behavior.groupby(['subject', 'session'], group_keys=False)['sd'].apply(lambda d: d>d.quantile()).map({True:'High neural uncertainty', False:'Low neural uncertainty'})

    return pupil, saccades, blinks, behavior, events

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(args.model_label, args.bids_folder)