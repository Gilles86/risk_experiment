import re
from tqdm import tqdm
import os
import os.path as op
import argparse
from fit_pupil_model import get_data
import pandas as pd
import nideconv
import warnings

def main(model_label, bids_folder):


    if not model_label == 'n1_n2_n':
        raise NotImplementedError

    target_dir = op.join(bids_folder, 'derivatives', 'pupil', f'model-{model_label}')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    pupil, saccades,  blinks, behavior, events = get_data(model_label, bids_folder)
    onsets = pd.concat((blinks, saccades), keys=['blink', 'saccade'], names=['event_type']).reset_index('event_type')[['event_type', 'onset']]

    n1 = events[events.event == 'n1'].join(behavior.reset_index('trial_nr')[['n1', 'trial_nr']]).rename(columns={'n1':'n'})[['onset', 'n', 'trial_nr']]
    n1['event_type'] = 'n1'
    n2 = events[events.event == 'n2'][['onset',]].join(behavior[['n2']]).rename(columns={'n2':'n'})[['onset', 'n']]
    n2['event_type'] = 'n2'

    n1_trials = events[events.event == 'n1'].join(behavior[[]])[['trial_type', 'onset']]
    n1_trials_pre = n1_trials.copy()
    n1_trials_post = n1_trials.copy()

    n1_trials_pre['event_type'] = n1_trials['trial_type'].apply(lambda x: f'{x}_pre')
    n1_trials_post['event_type'] = n1_trials['trial_type'].apply(lambda x: f'{x}_post')

    n2_trials = events[events.event == 'n2'][['onset',]].join(behavior[['n2', 'rt']])
    n2_trials['event_type'] = n2_trials.index.get_level_values('trial_nr').to_series(index=n2_trials.index).apply(lambda x: f'trial_{x:03d}_n2')
    n2_trials['onset'] += n2_trials['rt'].fillna(0.0)

    onsets = pd.concat((onsets, n1, n2, n1_trials_pre, n1_trials_post, n2_trials))
    onsets.drop('trial_type', axis=1, inplace=True)

    grf = nideconv.GroupResponseFitter(pupil['pupil'], onsets=onsets, input_sample_rate=10)
    grf.add_event('saccade', interval=[-0.5, 4.5], basis_set='dct', n_regressors=12)
    grf.add_event('blink', interval=[-0.5, 4.5], basis_set='dct', n_regressors=12)
    grf.add_event('n1', interval=[-3.5, 6.5], covariates='n', basis_set='dct', n_regressors=12)
    grf.add_event('n2', interval=[-3.5, 6.5], covariates='n', basis_set='dct', n_regressors=12)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for trial in tqdm(range(1, 193)):
            grf.add_event(f'trial_{trial:03d}_n1_pre', interval=[-1.3, -.8], basis_set='fir', n_regressors=1)
            grf.add_event(f'trial_{trial:03d}_n1_post', interval=[0.5, 4.0], basis_set='fir', n_regressors=1)
            grf.add_event(f'trial_{trial:03d}_n2', interval=[-1., 1.5], basis_set='fir', n_regressors=1)


    print('Fitting model...')
    grf.fit()
    print('Done fitting...')

    def extract_trial_betas(rf):
        d = rf.betas
        mask = d.index.get_level_values('event type').map(lambda x: x.startswith('trial_'))
        d = d.loc[mask]
        return d

    betas = []
    for ix, element in grf.concat_response_fitters.iteritems():
        betas.append(extract_trial_betas(element).droplevel([-1, -2]))

    betas = pd.concat(betas, keys=grf.concat_response_fitters.index).reset_index('event type')

    reg = re.compile('trial_(?P<trial_nr>[0-9]+)_(?P<event>n1|n2)_?(?P<prepost>pre|post)?')
    betas['trial_nr'] = betas['event type'].map(lambda x: reg.match(x).group(1)).astype(int)
    betas['prepost'] = betas['event type'].map(lambda x: reg.match(x).group(3)).apply(lambda x: 'post' if x is None else x)
    betas['event type'] = betas['event type'].map(lambda x: reg.match(x).group(2))
    betas = betas.set_index(['trial_nr', 'event type', 'prepost'], append=True)

    tmp = betas.unstack(['event type', 'prepost'])
    tmp = (tmp[('pupil', 'n1', 'post')] - tmp[('pupil', 'n1', 'pre')]).to_frame('pupil')
    tmp['event type'], tmp['prepost'] = 'n1', 'post-pre'
    tmp = tmp.set_index(['event type', 'prepost'], append=True)
    betas = pd.concat((betas, tmp)).sort_index()

    betas.to_csv(op.join(target_dir, 'pupil_pre_post12.tsv'), sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)
