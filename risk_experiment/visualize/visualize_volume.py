import numpy as np
import os.path as op
from nilearn import surface, image
import cortex
import numpy as np
from risk_experiment.utils.argparse import run_main, make_default_parser
from utils import _load_parameters

sourcedata = '/data/ds-risk'
subject = '32'
session = '7t1'
thr = .15

pc_subject = f'sub-{subject}'

parameter = 'r2'
vol_data = op.join(sourcedata, 'derivatives', 'encoding_model.volume', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_desc-{parameter}.optim_space-T1w_parameter.nii.gz')


parameter = 'mu'
vol_mu = op.join(sourcedata, 'derivatives', 'encoding_model.volume', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_desc-{parameter}.optim_space-T1w_parameter.nii.gz')


bold = image.load_img(op.join(sourcedata, 'derivatives', 'fmriprep', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-mapper_run-1_space-T1w_boldref.nii.gz'))

transform = cortex.xfm.Transform(np.identity(4), bold)
transform.save(pc_subject, 'bold')

r2_vol = cortex.Volume(vol_data, pc_subject, 'bold', vmin=0, vmax=.5)
vol_mu = cortex.Volume(vol_mu, pc_subject, 'bold', vmin=0, vmax=.5)

r2_surf = _load_parameters(subject, session, 'r2.optim', sourcedata)



ds = cortex.Dataset(r2=r2_vol, r2_surf=r2_surf, vol_mu=vol_mu)
cortex.webshow(ds)



