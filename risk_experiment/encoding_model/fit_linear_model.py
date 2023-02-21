import argparse
import pandas as pd
from braincoder.models import LinearModel
from braincoder.optimize import ParameterFitter, WeightFitter
from braincoder.utils import get_rsq
from risk_experiment.utils import get_target_dir
from nilearn.input_data import NiftiMasker
from nilearn import image

import os.path as op
import numpy as np

def main(subject, session, bids_folder='/data/ds-risk', smoothed=False,
        retroicor=False,
        denoise=False, 
        expected_value=False):

    key = 'glm_stim1'
    target_dir = 'encoding_model.lm'

    if expected_value:
        target_dir += '.ev'


    if denoise:
        key += '.denoise'
        target_dir += '.denoise'

    if (retroicor) and (not denoise):
        raise Exception("When not using GLMSingle RETROICOR is *always* used!")

    if retroicor:
        key += '.retroicor'
        target_dir += '.retroicor'

    if smoothed:
        key += '.smoothed'
        target_dir += '.smoothed'

    target_dir = get_target_dir(subject, session, bids_folder, target_dir)

    paradigm = [pd.read_csv(op.join(bids_folder, f'sub-{subject}', f'ses-{session}',
                                'func', f'sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'), sep='\t')
                for run in range(1, 9)]
    paradigm = pd.concat(paradigm, keys=range(1,9), names=['run'])
    paradigm['log(n1)'] = np.log(paradigm['n1'])
    paradigm['log(n2)'] = np.log(paradigm['n2'])

    paradigm = paradigm[paradigm.trial_type == 'stimulus 1'].set_index('trial_nr')

    if expected_value:
        paradigm['ev1'] = paradigm['prob1'] * paradigm['n1']
        paradigm = paradigm[['ev1']].astype(np.float32)
    else:
        paradigm = paradigm[['n1']].astype(np.float32)

    paradigm['intercept'] = np.float32(1.0)

    model = LinearModel()

    mask = op.join(bids_folder, 'derivatives', f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')

    masker = NiftiMasker(mask_img=mask)

    data_folder = op.join(bids_folder, 'derivatives', key,
                                                f'sub-{subject}', f'ses-{session}', 'func')
    data = op.join(data_folder, f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims1_pe.nii.gz')
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)

    optimizer = WeightFitter(model, parameters=None, data=data.astype(np.float32), paradigm=paradigm.astype(np.float32))

    weights = optimizer.fit()

    pred = model.predict(paradigm=paradigm, weights=weights)
    r2 = get_rsq(data, pred)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.optim_space-T1w_pars.nii.gz')
    masker.inverse_transform(r2).to_filename(target_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--retroicor', action='store_true')
    parser.add_argument('--expected_value', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder, smoothed=args.smoothed,
            denoise=args.denoise, retroicor=args.retroicor,
            expected_value=args.expected_value)
