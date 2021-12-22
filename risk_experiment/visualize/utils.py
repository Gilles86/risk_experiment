import cortex
from nilearn import surface
import numpy as np
import os.path as op

def _load_parameters(subject, session, par, sourcedata, space='fsnative', smoothed=True, concatenated=False, alpha=None):


    d_name = 'encoding_model'
    if smoothed:
        d_name += '.smoothed'
    if concatenated:
        d_name += '.concatenated'

    dir_ = op.join(sourcedata, 'derivatives', d_name, f'sub-{subject}',
            f'ses-{session}', 'func')

    print(dir_)

    par_l = op.join(dir_, f'sub-{subject}_ses-{session}_desc-{par}_space-{space}_hemi-L.func.gii')
    par_r = op.join(dir_, f'sub-{subject}_ses-{session}_desc-{par}_space-{space}_hemi-R.func.gii')

    if op.exists(par_l):
        par_l = surface.load_surf_data(par_l).T
        par_r = surface.load_surf_data(par_r).T

        par = np.concatenate((par_l, par_r))

        if space == 'fsnative':
            fs_subject = f'sub-{subject}'
        else:
            fs_subject = space

        return cortex.Vertex(par, fs_subject, alpha=alpha)
    else:
        print(f'Could not find {par_l}')
        return None

