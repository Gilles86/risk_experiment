import re
from tqdm import tqdm
import os
import os.path as op
import argparse
from fit_glm_model import get_data
import pandas as pd
import nideconv

def main(model_label, roi, bids_folder, session='7t2'):

    if not model_label == 'n1_n2_n':
        raise NotImplementedError

    target_dir = op.join(bids_folder, 'derivatives', 'roi_analysis', f'model-{model_label}', roi)

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    ts, events, behavior, confounds  = get_data(model_label, roi, bids_folder, session)

    n1 = events[events.event == 'n1'].join(behavior[['n1']]).rename(columns={'n1':'n'})[['onset', 'n']]
    n1['event_type'] = 'n1'
    n2 = events[events.event == 'n2'][['onset']].join(behavior[['n2']]).rename(columns={'n2':'n'})[['onset', 'n']]
    n2['event_type'] = 'n2'

    onsets = pd.concat((n1, n2))

    trial_regressors =events[events.event == 'n1'].join(behavior[[]])[['trial_type', 'onset']]
    trial_regressors['event_type'] = trial_regressors['trial_type'].map(lambda x: x[:-3])
    trial_regressors.drop('trial_type', axis=1, inplace=True)

    onsets = pd.concat((onsets, n1, n2, trial_regressors))

    grf = nideconv.GroupResponseFitter(ts, onsets=onsets, input_sample_rate=1./2.3, confounds=confounds, concatenate_runs=False)
    grf.add_event('n1', interval=[0.0, 2.3*8], basis_set='dct', n_regressors=8, covariates='n')
    grf.add_event('n2', interval=[0.0, 2.3*8], basis_set='dct', n_regressors=8, covariates='n')

    for trial in tqdm(trial_regressors['event_type'].unique()):
        grf.add_event(trial, interval=[-2.3*3, 0.0], basis_set='fir', n_regressors=1)


    grf.fit()

    def extract_trial_betas(rf):
        d = rf.betas
        mask = d.index.get_level_values('event type').map(lambda x: x.startswith('trial_'))
        d = d.loc[mask]
        return d

    betas = []
    for ix, element in grf.response_fitters.iteritems():
        betas.append(extract_trial_betas(element).droplevel([-1, -2]))

    betas = pd.concat(betas, keys=grf.response_fitters.index).reset_index('event type')

    reg = re.compile('trial_(?P<trial_nr>[0-9]+)')
    betas['trial_nr'] = betas['event type'].map(lambda x: reg.match(x).group(1)).astype(int)
    betas = betas.set_index('trial_nr', append=True).drop('event type', axis=1)

    betas.to_csv(op.join(target_dir, f'ses-{session}_pre_stim_baseline.tsv'), sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('roi', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(args.model_label, args.roi, bids_folder=args.bids_folder)
