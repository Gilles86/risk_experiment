import os.path as op
from nilearn import surface
import cortex
import numpy as np
from risk_experiment.utils.argparse import run_main, make_default_parser
from utils import _load_parameters

sourcedata = '/data/ds-risk'
subject = '05'
session = '7t1'
thr = .15

parameter = 'r2'
vol_data = op.join(sourcedata, 'derivatives', 'encoding_model.volume', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_desc-{parameter}.optim_space-T1w_parameter.nii.gz')

r2_vol = cortex.Volume(vol_data, f'sub-{subject}', 'bold', vmin=0, vmax=.5)

r2_surf = _load_parameters(subject, session, 'r2.optim', sourcedata)


ds = cortex.Dataset(r2=r2_vol, r2_surf=r2_surf)
cortex.webshow(ds)



