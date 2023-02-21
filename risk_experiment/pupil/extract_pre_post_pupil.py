import re
from tqdm import tqdm
import os
import os.path as op
import argparse
from fit_pupil_model import get_data
import pandas as pd
import nideconv

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

    trial_regressors =events[events.event == 'n1'].join(behavior[[]])[['trial_type', 'onset']]
    trial_regressors['event_type'] = trial_regressors['trial_type'].map(lambda x: x[:-3])
    trial_regressors.drop('trial_type', axis=1, inplace=True)

    onsets = pd.concat((onsets, n1, n2, trial_regressors))

    grf = nideconv.GroupResponseFitter(pupil['pupil'], onsets=onsets, input_sample_rate=10)
    grf.add_event('saccade', interval=[-0.5, 4.5], basis_set='dct', n_regressors=12)
    grf.add_event('blink', interval=[-0.5, 4.5], basis_set='dct', n_regressors=12)
    grf.add_event('n1', interval=[-3.5, 6.5], basis_set='dct', n_regressors=12)
    grf.add_event('n2', interval=[-3.5, 6.5], basis_set='dct', n_regressors=12)

    for trial in tqdm(trial_regressors['event_type'].unique()):
        grf.add_event(trial, interval=[-2.5, -.5], basis_set='fir', n_regressors=1)


    grf.fit()

    def extract_trial_betas(rf):
        d = rf.betas
        mask = d.index.get_level_values('event type').map(lambda x: x.startswith('trial_'))
        d = d.loc[mask]
        return d

    betas = []
    for ix, element in grf.concat_response_fitters.iteritems():
        betas.append(extract_trial_betas(element).droplevel([-1, -2]))

    betas = pd.concat(betas, keys=grf.concat_response_fitters.index.astype(int)).reset_index('event type')

    reg = re.compile('trial_(?P<trial_nr>[0-9]+)')
    betas['trial_nr'] = betas['event type'].map(lambda x: reg.match(x).group(1)).astype(int)
    betas = betas.set_index('trial_nr', append=True).drop('event type', axis=1)

    betas.to_csv(op.join(target_dir, 'pre_stim_baseline.tsv'), sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)
