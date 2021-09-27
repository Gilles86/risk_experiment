import os
import os.path as op
import argparse
import pandas as pd
import numpy as np
from nilearn import image
from risk_experiment.utils import get_mapper_paradigm, get_target_dir, get_volume_data
from risk_experiment.utils import get_behavior, get_fmriprep_confounds, get_retroicor_confounds, get_tr, get_mapper_response_hrf, get_runs, get_brain_mask
from risk_experiment.utils.math import psc
from nilearn.input_data import NiftiMasker
from braincoder.models import GaussianPRFWithHRF
from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter

def main(subject, session, bids_folder, smoothed=True, concatenate=False, space='T1w'):

    target_dir = 'encoding_model.volume'

    target_dir = get_target_dir(subject, session, bids_folder, target_dir)

    # Create confounds
    fmriprep_confounds = get_fmriprep_confounds(subject, session, bids_folder)
    retroicor_confounds = get_retroicor_confounds(subject, session, bids_folder)
    response_hrf = get_mapper_response_hrf(subject, session, bids_folder)

    confounds = pd.concat((fmriprep_confounds, retroicor_confounds,
        response_hrf), axis=1)

    paradigm = get_mapper_paradigm(subject, session, bids_folder)

    images = []


    psc_dir = op.join(bids_folder, 'derivatives', 'psc', f'sub-{subject}', f'ses-{session}', 'func')

    if not op.exists(psc_dir):
        os.makedirs(psc_dir)

    masks = []

    runs = get_runs(subject, session)

    for run in runs:
        print(f'cleaning run {run}')
        d = get_volume_data(subject, session, run, bids_folder, space=space)

        d = psc(d)

        d_cleaned = image.clean_img(d, confounds=confounds.loc[run].values, standardize=False, detrend=False, ensure_finite=True)

        d_cleaned.to_filename(op.join(psc_dir, f'sub-{subject}_ses-{session}_task-mapper_run-{run}_desc-psc_bold.nii.gz'))

        images.append(d_cleaned)
        masks.append(get_brain_mask(subject, session, run, bids_folder))

    conjunct_mask = image.math_img('mask.sum(-1).astype(np.bool)', mask=image.concat_imgs(masks))

    masker = NiftiMasker(conjunct_mask)

    data = [pd.DataFrame(masker.fit_transform(im), index=paradigm.index)  for im in images]
    data = pd.concat(data, keys=runs, names=['run'])
    data.columns.name = 'voxel'

    mean_data =data.groupby('time').mean()

    mean_image = masker.inverse_transform(mean_data)
    mean_target_dir = get_target_dir(subject, session, bids_folder, 'mean_clean_volumes')
    mean_image.to_filename(op.join(mean_target_dir, f'sub-{subject}_ses-{session}_task-mapper_desc-meanedcleaned_bold.nii.gz'))

    hrf_model = SPMHRFModel(tr=get_tr(subject, session), time_length=20)
    model = GaussianPRFWithHRF(hrf_model=hrf_model)

    # # SET UP GRID
    mus = np.linspace(0, np.log(80), 20, dtype=np.float32)
    sds = np.linspace(.01, 2, 15, dtype=np.float32)
    amplitudes = np.linspace(1e-6, 10, 10, dtype=np.float32)
    baselines = np.array([0.0], dtype=np.float32)

    optimizer = ParameterFitter(model, mean_data, paradigm)

    grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines)

    r2 = optimizer.get_rsq(grid_parameters)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.grid_space-{space}_parameter.nii.gz')
    masker.inverse_transform(r2).to_filename(target_fn)

    for par, values in grid_parameters.T.iterrows():
        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.grid_space-{space}_parameter.nii.gz')
        masker.inverse_transform(values).to_filename(target_fn)

    optimizer.fit(init_pars=grid_parameters, learning_rate=.1, store_intermediate_parameters=False, max_n_iterations=5000)

    r2 = optimizer.r2

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.optim_space-{space}_parameter.nii.gz')
    masker.inverse_transform(r2).to_filename(target_fn)

    for par, values in grid_parameters.T.iterrows():
        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim_space-{space}_parameter.nii.gz')
        masker.inverse_transform(values).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--concatenate', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder, concatenate=args.concatenate)
