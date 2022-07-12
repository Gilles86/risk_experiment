from risk_experiment.utils.data import write_gifti
import os.path as op
from nilearn import surface
import numpy as np
import glob
import pandas as pd
import re
from tqdm import tqdm
import numpy as np

sourcedata='/data/ds-risk'


for smoothed in [True]:
    folder = 'encoding_model'

    if smoothed:
        folder += '.smoothed'

    fns = glob.glob(op.join(sourcedata, 'derivatives', folder, f'sub-*',
                   'ses-*', 'func', '*volume.optim*space-fsaverage_hemi-*.func.gii'))
    # fns += glob.glob(op.join(sourcedata, 'derivatives', folder, f'sub-*',
                   # 'ses-7t1', 'func', '*volume.optim*space-fsaverage_hemi-*.func.gii'))

    reg = re.compile('.*/sub-(?P<subject>[0-9]+)_ses-(?P<session>3t1|7t1|3t2|7t2)_desc-(?P<parameter>[0-9a-z]+)\.volume\.optim_space-.+_hemi-(?P<hemi>L|R)\.func\.gii')
    keys = []
    data = []

    # parameters = ['mu', 'r2']

    for fn in tqdm(fns):

        d = reg.match(fn).groupdict()

        # if (d['parameter'] in parameters):
        keys.append(reg.match(fn).groupdict())
        data.append(surface.load_surf_data(fn))


    keys = pd.MultiIndex.from_frame(pd.DataFrame(keys))
    data = pd.DataFrame(data, index=keys).unstack('hemi')
    data = data.reorder_levels([1, 0], 1).sort_index(axis=1)

    for session, d in data.groupby(['session']):

        weighted_mu = d.xs('mu', level='parameter') * d.xs('r2', level='parameter')
        weighted_mu = weighted_mu.sum() / d.xs('r2', level='parameter').sum()

        write_gifti('02', session, sourcedata, 'fsaverage', weighted_mu,
                op.join(sourcedata, 'derivatives', folder, f'group_ses-{session}_desc-weightedmu.volume_hemi-_hemi__mean.gii'.replace('_hemi_', '{hemi}')))

    for (session, par), d in data.groupby(['session', 'parameter']):
        print(d)

        d = d.mean()

        write_gifti('02', session, sourcedata, 'fsaverage', d,
                op.join(sourcedata, 'derivatives', folder, f'group_ses-{session}_desc-{par}.volume_hemi-_hemi__mean.gii'.replace('_hemi_', '{hemi}')))
