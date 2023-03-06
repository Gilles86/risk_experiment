import cortex
from risk_experiment.utils import Subject, get_all_subject_ids, get_all_subjects
import os.path as op
from nilearn import image
import pandas as pd
from itertools import product

bids_folder = '/data/ds-risk'


subjects = get_all_subject_ids()


subjects = get_all_subjects(bids_folder)

keys = ['encoding_model.denoise.smoothed.natural_space', 'encoding_model.ev.denoise.smoothed.natural_space',
            'encoding_model.lm.ev.denoise.smoothed']

pars = []
for key, session in product(keys, ['3t2', '7t2']):
    p = [sub.get_prf_parameters_surf(session, space='fsaverage', key=key, parameters=['r2']) for sub in subjects]
    p = pd.concat(p, keys=[(sub.subject, session, key) for sub in subjects], names=['subject', 'session', 'key'])
    pars.append(p)

pars = pd.concat(pars)

d = {}
for key, session in product(keys, ['3t2', '7t2']):
    d[f'{session}.{key}'] = cortex.Vertex(pars.xs(key, 0, 'key').xs(session, 0, 'session').groupby(['hemi', 'vertex']).mean().values.ravel(), 'fsaverage',
    vmin=0.025, vmax=0.1)

# d = {}
# d['3t2.n.vs.ev'] = cortex.Vertex2D(pars.xs('encoding_model.denoise.smoothed.natural_space', 0, 'key').xs('3t2', 0, 'session').groupby(['hemi', 'vertex']).mean().values.ravel(),
#                                    pars.xs('encoding_model.ev.denoise.smoothed.natural_space', 0, 'key').xs('3t2', 0, 'session').groupby(['hemi', 'vertex']).mean().values.ravel(), 
#                                    'fsaverage', vmin=0.00, vmax=0.12, vmin2=0.00, vmax2=0.12,
#                                    cmap='PU_RdBu_covar_alpha')
# d['7t2.n.vs.ev'] = cortex.Vertex2D(pars.xs('encoding_model.denoise.smoothed.natural_space', 0, 'key').xs('7t2', 0, 'session').groupby(['hemi', 'vertex']).mean().values.ravel(),
#                                    pars.xs('encoding_model.ev.denoise.smoothed.natural_space', 0, 'key').xs('7t2', 0, 'session').groupby(['hemi', 'vertex']).mean().values.ravel(), 
#                                    'fsaverage', vmin=0.05, vmax=0.12, vmin2=0.05, vmax2=0.12,
#                                    cmap='PU_RdBu_covar_alpha')
ds = cortex.Dataset(**d)

cortex.webshow(ds)