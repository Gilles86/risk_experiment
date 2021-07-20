import os.path as op
from nilearn import surface
import cortex
import numpy as np
from risk_experiment.utils.argparse import run_main, make_default_parser
import glob
from itertools import product
import pandas as pd
import re
from matplotlib import colors, cm

sourcedata='/data2/ds-risk'

# def main(sourcedata):
dir_ = op.join(sourcedata, 'derivatives', 'encoding_model.smoothed', f'sub-*',
               'ses-{session}', 'func')

reg = re.compile('.*/sub-(?P<subject>[a-z0-9]+)_ses-.+_desc-.+_space-.+_hemi-.+.func.gii')
keys = []
data = []

parameters = ['r2.optim', 'mu.optim']

for session, hemi, par in product(['3t1', '7t1'], ['L', 'R'], parameters):
    fns = glob.glob(op.join(dir_.format(session=session),
                       f'sub-*_ses-{session}_desc-{par}_space-fsaverage_hemi-{hemi}.func.gii'))

    for fn in fns:
        subject = reg.match(fn).group(1)
        key = {'session':session, 'hemi':hemi, 'par':par, 'subject':subject}
        keys.append(key)
        data.append(surface.load_surf_data(fn))

keys = pd.MultiIndex.from_frame(pd.DataFrame(keys))
data = pd.DataFrame(data, index=keys).unstack('hemi')
data = data.reorder_levels([1, 0], 1).sort_index(axis=1)

d = {}
for p in parameters:
    d[f'{p}.3t'] = cortex.Vertex(data.loc[('3t1', p)].mean().values, 'fsaverage')
    d[f'{p}.7t'] = cortex.Vertex(data.loc[('7t1', p)].mean().values, 'fsaverage')


ix_ = pd.IndexSlice

def get_weighted_mu(d):    
    return (d.loc[ix_[:, 'mu.optim', :]] * data.loc[ix_[:, 'r2.optim', :]] / d.loc[ix_[:, 'r2.optim', :]].sum(0)).sum(0)


d['weighted_mu_3t'] = cortex.Vertex(get_weighted_mu(data.loc[['3t1']]).values, 'fsaverage')
d['weighted_mu_7t'] = cortex.Vertex(get_weighted_mu(data.loc[['7t1']]).values, 'fsaverage')


mu_min = 1.5
mu_max = 2.5
r2_thr = 0.11
mu_exp = np.exp(d['weighted_mu_7t'].data)
mu_min = 5.
mu_max = 15.

r2 = d['r2.optim.7t'].data 
alpha =  r2  / r2.max()
alpha[mu_exp > 20] = 0.0
# mu_exp[mu_exp > 20] = np.nan
mu_exp[r2 < r2_thr] = np.nan

mu_exp = np.clip((mu_exp - mu_min) / (mu_max - mu_min), 0, 1.)
red, green, blue = cm.hsv(mu_exp,)[:, :3].T

red = cortex.Vertex(red, 'fsaverage', vmin=0, vmax=1)
blue = cortex.Vertex(blue, 'fsaverage', vmin=0, vmax=1)
green = cortex.Vertex(green, 'fsaverage', vmin=0, vmax=1)
alpha = cortex.Vertex(alpha, 'fsaverage', vmin=0, vmax=1)

weighted_mu_rgb_v = cortex.VertexRGB(
    red=red, green=green, blue=blue, subject='fsaverage', alpha=alpha,
    vmin=0., vmax=1.)

d['mu_rgb'] = weighted_mu_rgb_v

# cortex.webshow(d['mu_rgb'])

ds = cortex.Dataset(**d)

cortex.webshow(ds)

# if __name__ == '__main__':
    # main(sourcedata='/data2/ds-risk')
