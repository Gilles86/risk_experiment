import os
import os.path as op
import argparse
import pandas as pd
import numpy as np
from nilearn import image

def main(subject, session, bids_folder, max_rt=1.0):

    sourcedata = op.join(bids_folder, 'sourcedata')

    target_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    if session[-1] == '1':

        for run in range(1, 5):

            nii = op.join(target_dir, f'sub-{subject}_ses-{session}_task-mapper_run-{run}_bold.nii.gz')
            n_volumes = image.load_img(nii).shape[-1]


            behavior = pd.read_table(op.join(sourcedata, f'sub-{subject}/behavior/ses-{session}/sub-{subject}_ses-{session}_task-mapper_run-{run}_events.tsv'))

            pulses = behavior[behavior.event_type == 'pulse'][['trial_nr', 'onset']]

            pulses['ipi'] = pulses['onset'].diff()
            pulses = pulses[((pulses['ipi'] > .2) & (pulses['ipi'] < 5.)) | pulses.ipi.isnull()]
            pulses = pulses.set_index(np.arange(1, n_volumes+1))[['trial_nr', 'onset']]
            t0 = pulses.loc[1, 'onset']

            targets = behavior[(behavior.color == 1.0) & (behavior.event_type == 'stim') & (behavior.n_dots > 0.0) & (behavior['phase'] % 2 == 0)]
            targets = targets.pivot_table(index=['trial_nr', 'phase'], values='onset')
            targets['onset'] = targets['onset'] - t0

            responses = behavior[behavior['event_type']  == 'response']
            responses['onset'] -= t0
            responses = responses.pivot_table(index=['trial_nr', 'phase'], values='onset', aggfunc='min')

            stim_response = (responses[['onset']].T.values - targets[['onset']].values)
            stim_response = pd.DataFrame(stim_response, index=targets.index, columns=responses.index)
            stim_response[(stim_response < 0) |(stim_response > max_rt)] = np.nan
            targets['rt'] = stim_response.min(1)
            targets['responded'] = ~targets['rt'].isnull()

            targets['isi'] = targets['onset'].diff()

            targets['hazard1'] = get_hazard(targets['isi'])
            targets['hazard2'] = get_hazard(targets['isi'], use_cut=True)

            all_stimuli = behavior[(behavior.event_type == 'stim') & (~behavior.n_dots.isnull())]
            all_stimuli['onset'] -= t0
            start_dots = all_stimuli.pivot_table(index=['trial_nr'], values=['onset'], aggfunc=np.min)
            end_dots = all_stimuli.pivot_table(index=['trial_nr'], values=['onset'], aggfunc=np.max)
            n_dots = all_stimuli.pivot_table(index=['trial_nr'], values=['n_dots'], aggfunc=np.mean)

            all_stimuli = pd.concat([start_dots, end_dots, n_dots], axis=1)
            all_stimuli.columns = ['onset', 'end', 'n_dots']
            all_stimuli['duration'] = all_stimuli['end'] - all_stimuli['onset']
            all_stimuli = all_stimuli.drop('end', 1)


            all_stimuli['trial_type'] = 'stimulation'
            responses['trial_type'] = 'response'
            targets['trial_type'] = 'targets'

            events = pd.concat((all_stimuli, responses, targets), ignore_index=True)


            fn = op.join(target_dir, f'sub-{subject}_ses-{session}_task-mapper_run-{run}_events.tsv')
            events.to_csv(fn, index=False, sep='\t')

    else:



        for run in range(1, 9):
            nii = op.join(target_dir, f'sub-{subject}_ses-{session}_task-task_run-{run}_bold.nii.gz')
            n_volumes = image.load_img(nii).shape[-1]


            behavior = pd.read_table(op.join(sourcedata, f'sub-{subject}/behavior/ses-{session}/sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'))
            behavior['trial_nr'] = behavior['trial_nr'].astype(np.int)

            print(behavior)

            pulses = behavior[behavior.event_type == 'pulse'][['trial_nr', 'onset']]

            pulses['ipi'] = pulses['onset'].diff()
            pulses = pulses[((pulses['ipi'] > .2) & (pulses['ipi'] < 5.)) | pulses.ipi.isnull()]
            pulses = pulses.set_index(np.arange(1, n_volumes+1))[['trial_nr', 'onset']]
            t0 = pulses.loc[1, 'onset']


            stim1 = behavior[(behavior['event_type'] == 'stim') & (behavior['phase'] == 4)]
            stim1['n'] = stim1['n1']
            stim1['onset'] -= t0
            stim1['event_type'] = 'stimulus 1'


            stim2 = behavior[(behavior['event_type'] == 'stim') & (behavior['phase'] == 8)]
            stim2['n'] = stim2['n2']
            stim2['onset'] -= t0
            stim2['event_type'] = 'stimulus 2'


            choice = behavior[(behavior['event_type'] == 'choice')]
            choice['onset'] -= t0
            choice['event_type'] = 'choice'


            certainty = behavior[(behavior['event_type'] == 'certainty')]
            certainty['onset'] -= t0
            certainty['event_type'] = 'certainty'
            certainty['choice'] = certainty['certainty'].astype(int)


            events = pd.concat((stim1, stim2, choice, certainty)).sort_index().reset_index(drop=True)
            # result['choice'] = result['choice'].astype(int)
            events = events[['trial_nr', 'onset', 'event_type', 'prob1', 'prob2', 'n1', 'n2', 'choice']]

            fn = op.join(target_dir, f'sub-{subject}_ses-{session}_task-task-{run}_events.tsv')
            events.to_csv(fn, index=False, sep='\t')


def get_hazard(x, s=1.0, loc=0.0, scale=10, cut=30, use_cut=False):
    import scipy.stats as ss
    
    x = x / .7

    dist = ss.lognorm(s, loc, scale)
    
    if use_cut:
        sf = lambda x: 1 - (dist.cdf(x) / dist.cdf(cut))
    else:
        sf = dist.sf

    return np.clip(dist.pdf(x) / sf(x), 0, np.inf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder)
