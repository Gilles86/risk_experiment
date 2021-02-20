import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
import argparse
import os
import os.path as op
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils import get_behavior, get_fmriprep_confounds, get_retroicor_confounds, get_tr, get_mapper_response_hrf
from utils import get_surf_data, get_mapper_paradigm, write_gifti, get_target_dir
import pandas as pd

from nilearn import signal

from braincoder.models import GaussianPRFWithHRF, GaussianPRF
from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterOptimizer

def main(subject, session, sourcedata):

    print('yo')
    target_dir = get_target_dir(subject, session, sourcedata, 'encoding_model')

    # Create confounds
    fmriprep_confounds = get_fmriprep_confounds(subject, session, sourcedata)
    retroicor_confounds = get_retroicor_confounds(subject, session, sourcedata)
    response_hrf = get_mapper_response_hrf(subject, session, sourcedata)

    confounds = pd.concat((fmriprep_confounds, retroicor_confounds,
        response_hrf), axis=1)

    # Get surface data
    surf = get_surf_data(subject, session, sourcedata)


    # # clean surface data
    surf_cleaned = pd.DataFrame(None, columns=surf.columns)
    for run, d in surf.groupby(['run']):
        d_cleaned = signal.clean(d.values, confounds=confounds.loc[run].values, standardize='psc')
        surf_cleaned = surf_cleaned.append(pd.DataFrame(d_cleaned, columns=d.columns))
    
    surf_cleaned.index = surf.index

    avg_data = surf_cleaned.groupby(level=1, axis=0).mean()

    paradigm = get_mapper_paradigm(subject, session, sourcedata)


    hrf_model = SPMHRFModel(tr=get_tr(subject, session), time_length=20)
    model = GaussianPRFWithHRF(hrf_model=hrf_model)

    # avg_data = avg_data.loc[:, avg_data.var() != 0].astype(np.float32) 

    # SET UP GRID
    mus = np.linspace(0, np.log(80), 10, dtype=np.float32)
    sds = np.linspace(0, 2, 10, dtype=np.float32)
    amplitudes = np.linspace(0, 1, 10, dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    optimizer = ParameterOptimizer(model, avg_data, paradigm)
    grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines)

    r2 = optimizer.get_r2(grid_parameters)
    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.grid_space-fsnative') + '_hemi-{hemi}.func.gii'

    write_gifti(subject, session, sourcedata, 'fsnative', r2, target_fn)

    for par, values in grid_parameters.T.iterrows():
        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.grid_space-fsnative') + '_hemi-{hemi}.func.gii'
        write_gifti(subject, session, sourcedata, 'fsnative', values, target_fn)

    optimizer.fit(init_pars=grid_parameters, learning_rate=0.05, store_intermediate_parameters=False)
    print(optimizer.r2)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.optim_space-fsnative') + '_hemi-{hemi}.func.gii'
    write_gifti(subject, session, sourcedata, 'fsnative', optimizer.r2, target_fn)

    for par, values in optimizer.estimated_parameters.T.iterrows():
        print(values)
        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim_space-fsnative') + '_hemi-{hemi}.func.gii'
        write_gifti(subject, session, sourcedata, 'fsnative', values, target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--sourcedata', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, sourcedata=args.sourcedata)
