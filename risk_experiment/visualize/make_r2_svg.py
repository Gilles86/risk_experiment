import numpy as np
import os.path as op
import cortex
from nilearn import surface

folder = 'encoding_model'

sourcedata = '/data/ds-risk'

par = 'r2'
session = '3t2'

r2_l = surface.load_surf_data(op.join(sourcedata, 'derivatives',
                                                   folder, f'group_ses-{session}_desc-{par}.volume_hemi-L_mean.gii'))
r2_r = surface.load_surf_data(op.join(sourcedata, 'derivatives',
                                                   folder, f'group_ses-{session}_desc-{par}.volume_hemi-R_mean.gii'))

r2 = np.concatenate((r2_l, r2_r))

r2[r2 <0.025] = np.nan

r2 = cortex.Vertex(r2, 'fsaverage', cmap='hot', vmin=0.0, vmax=.05)

cortex.utils.add_roi(r2, 'r2_3t2_group', open_inkscape=True)
