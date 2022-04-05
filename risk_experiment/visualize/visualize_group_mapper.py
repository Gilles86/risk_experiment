import matplotlib.pyplot as plt
import os.path as op
from nilearn import surface
import cortex
import numpy as np
from risk_experiment.utils.argparse import run_main, make_default_parser
import glob
from itertools import product
import pandas as pd
import re
from tqdm import tqdm
from pandas import IndexSlice as ix_
from utils import get_alpha_vertex
from tqdm.contrib.itertools import product

sourcedata = '/data/ds-risk'


smoothed = False


parameters = ['mu', 'r2', 'weightedmu']
sessions = ['3t1', '7t1', '3t2', '7t2']
hemis = ['L', 'R']

data = []
keys = []


for smoothed in [False]:
    folder = 'encoding_model'

    if smoothed:
        folder += '.smoothed'

    for par, session, hemi in product(parameters, sessions, hemis):
        data.append(surface.load_surf_data(op.join(sourcedata, 'derivatives',
                                                   folder, f'group_ses-{session}_desc-{par}.volume_hemi-{hemi}_mean.gii')))
        keys.append({'smoothed': 'sm' if smoothed else 'usm', 'parameter': par, 'session': session,
                     'hemi': hemi})


data = pd.DataFrame(data, index=pd.MultiIndex.from_frame(pd.DataFrame(keys)))
data = data.unstack('hemi').reorder_levels([1, 0], 1).sort_index(axis=1)

mu_natural = np.exp(data.loc[ix_[:, ['mu'], :], :])
mu_natural.index = mu_natural.index.set_levels(['mu_natural'], 1)

data = pd.concat((data, mu_natural))


d = {}

counter = 0

for smoothed, session in product(['usm'], ['3t1', '3t2', '7t1', '7t2']):
    # alpha = np.copy(d[f'r2.{session}.{smoothed}'].data)
    alpha = data.loc[(smoothed, 'r2', session)]

    if session.endswith('1'):
        thr = 0.035
        alpha_ = np.clip((alpha - thr)/(thr/2.), .0, 1.0)
    else:
        thr = 0.03
        alpha_ = np.clip((alpha - thr)/(thr/2.), .0, 1.0)

    d[f'r2_.{session}.{smoothed}'] = get_alpha_vertex(alpha.values, alpha_, cmap='hot',
                                                      vmin=0.0, vmax=.15)



    mu = data.loc[(smoothed, 'mu', session)]
    d[f'mu.{session}.{smoothed}'] = get_alpha_vertex(
        mu.values, alpha_,
        vmax=np.log(40))

    mu_natural = data.loc[(smoothed, 'mu_natural', session)]
    d[f'mu_natural.{session}.{smoothed}'] = get_alpha_vertex(
        mu_natural.values, alpha_,
        vmin=5.0,
        vmax=40)

    weighted_mu = data.loc[(smoothed, 'weightedmu', session)]
    d[f'weighted_mu.{session}.{smoothed}'] = get_alpha_vertex(
        weighted_mu.values, alpha_,
        vmin=np.log(5.),
        vmax=np.log(40.))

ds = cortex.Dataset(**d)
cortex.webshow(ds)
