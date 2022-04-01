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

sourcedata = '/data/ds-risk'


smoothed = False


parameters = ['mu', 'r2']
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
for (smoothed, parameter, session), value in data.iterrows():

    counter += 1

    if parameter == 'r2':
        vmin, vmax = 0.0, 0.5
        cmap = 'magma'
    elif parameter == 'mu':
        vmin, vmax = np.log(5), np.log(28)
        cmap = 'nipy_spectral'
    elif parameter == 'mu_natural':
        vmin, vmax = 5, 80
        cmap = 'nipy_spectral'

    key = f'{parameter}.{session}.{smoothed}'
    d[key] = cortex.Vertex(
        value.values, 'fsaverage', vmin=vmin, vmax=vmax, cmap=cmap)


# for smoothed, session in product(['sm', 'usm'], ['3t1', '7t1', '3t2', '7t2']):
for smoothed, session in product(['usm'], ['3t1', '3t2', '7t1', '7t2']):
    alpha = np.copy(d[f'r2.{session}.{smoothed}'].data)
    if session.endswith('1'):
        d[f'r2_.{session}.{smoothed}'] = get_alpha_vertex(alpha, np.clip((alpha - 0.035)/0.0175, .0, 1.0), cmap='hot',
                                                          vmin=0.0, vmax=.15)
    else:
        d[f'r2_.{session}.{smoothed}'] = get_alpha_vertex(alpha, np.clip((alpha - 0.025)/0.0125, .0, 1.0), cmap='hot',
                                                          vmin=0.0, vmax=.15)

    # alpha -= 0.035
    alpha -= np.nanpercentile(alpha, 65)
    alpha = np.clip(alpha / np.percentile(alpha[alpha > 0.0], 50), 0, 1)
    alpha = alpha / alpha.max()
    # d[f'mu_.{session}.{smoothed}'] = get_alpha_vertex(
        # d[f'mu.{session}.{smoothed}'].data, alpha,
        # vmax=np.log(28))

ds = cortex.Dataset(**d)
cortex.webshow(ds)

x = np.linspace(0, 1, 101, True)
y = np.linspace(np.log(5), np.log(80), len(x), True)

plt.imshow(plt.cm.nipy_spectral(x)[np.newaxis, ...], aspect=10)


ns = [5, 7, 10, 14, 20, 28, 40, 56, 80]
plt.xticks((np.log(ns) - y[0]) / (y[-1] - y[0]) * len(x), ns)
plt.show()

# print(data)

# ix_ = pd.IndexSlice

# for p in parameters:
# d[f'{p}.3t'] = cortex.Vertex(data.loc[ix_[:, '3t1', p], :].mean().values, 'fsaverage')
# d[f'{p}.7t'] = cortex.Vertex(data.loc[ix_[:, '7t1', p], :].mean().values, 'fsaverage')


# d['mu.optim.3t.natural'] = cortex.Vertex(np.exp(data.loc[ix_[:, '3t1', 'mu'], :].mean(0)).values, 'fsaverage')
# d['mu.optim.7t.natural'] = cortex.Vertex(np.exp(data.loc[ix_[:, '7t1', 'mu'], :].mean(0)).values, 'fsaverage')

# def get_weighted_mu(d):
# return (d.loc[ix_[:, :, 'mu']] * d.loc[ix_[:, :, 'r2']] / d.loc[ix_[:, :, 'r2']].sum(0)).sum(0)


# d['weighted_mu_3t'] = cortex.Vertex(get_weighted_mu(data.loc[ix_[:, '3t1'], :]).values, 'fsaverage')
# d['weighted_mu_3t_natural'] = cortex.Vertex(np.exp(d['weighted_mu_3t'].data), 'fsaverage')
# d['weighted_mu_7t'] = cortex.Vertex(get_weighted_mu(data.loc[ix_[:, '7t1'], :]).values, 'fsaverage')
# d['weighted_mu_7t_natural'] = cortex.Vertex(np.exp(d['weighted_mu_7t'].data), 'fsaverage')


# mu_min = 1.5
# mu_max = 2.5
# r2_thr = 0.025
# mu = d['weighted_mu_7t'].data
# mu = d['mu.7t'].data
# mu_exp = np.exp(mu)
# mu_min = 5.0
# mu_max = 80.

# r2 = d['r2.7t'].data
# alpha =  r2 / np.percentile(r2, 95)
# alpha = np.clip(alpha, 0.0, 1.0)
# alpha[mu_exp > 80] = 0.0
# alpha = r2 - 0.05
# alpha = np.clip(alpha / np.percentile(alpha[alpha > 0.0], 80), 0, 1)

# alpha = alpha / alpha.max()

# mu_exp = np.clip((mu_exp - mu_min) / (mu_max - mu_min), 0, 1.)
# red, green, blue = cm.viridis(mu_exp,)[:, :3].T

# mu = np.clip((mu - np.log(mu_min)) / (np.log(mu_max) - np.log(mu_min)), 0, 1.)
# red, green, blue = cm.nipy_spectral(mu,)[:, :3].T

# # Get curvature
# curv = cortex.db.get_surfinfo('fsaverage')
# # Adjust curvature contrast / color. Alternately, you could work
# # with curv.data, maybe threshold it, and apply a color map.
# curv.data = np.sign(curv.data.data) * .25
# curv.vmin = -1
# curv.vmax = 1
# curv.cmap = 'gray'
# curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data]).astype(np.float32)

# vx_rgb = (np.vstack([red.data, green.data, blue.data]) * 255.).astype(np.float32)

# display_data = vx_rgb * alpha[np.newaxis, :] + curv_rgb * (1.-alpha[np.newaxis, :])

# d['mu_rgb'] = cortex.VertexRGB(*display_data.astype(np.uint8), 'fsaverage')


# # cortex.webshow(d['mu_rgb'])

# # ds = cortex.Dataset(**d)

# # cortex.webshow(ds)
# cortex.webshow(d['mu_rgb'])

# # if __name__ == '__main__':
# # main(sourcedata='/data2/ds-risk')
