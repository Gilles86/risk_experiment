import os.path as op
from nilearn import surface
import cortex
import numpy as np

sourcedata = '/data2/ds-risk'
subject = '10'
session = '3t1'
thr = .15


left, right = cortex.db.get_surf(subject, 'fiducial', merge=False)
left, right = cortex.Surface(*left), cortex.Surface(*right)


def smooth(data, *args, **kwargs):
    n_left_pts = len(left.pts)

    smoothed_data = np.zeros_like(data)
    smoothed_data[:n_left_pts] = left.smooth(data[:n_left_pts], *args, **kwargs)
    smoothed_data[n_left_pts:] = right.smooth(data[n_left_pts:], *args, **kwargs)

    return smoothed_data


def _load_parameters(subject, session, par, smoothed=False, concatenated=False, alpha=None):


    d_name = 'encoding_model'
    if smoothed:
        d_name += '.smoothed'
    if concatenated:
        d_name += '.concatenated'

    dir_ = op.join(sourcedata, 'derivatives', d_name, f'sub-{subject}',
            f'ses-{session}', 'func')

    print(dir_)

    par_l = op.join(dir_, f'sub-{subject}_ses-{session}_desc-{par}_space-fsnative_hemi-L.func.gii')
    par_r = op.join(dir_, f'sub-{subject}_ses-{session}_desc-{par}_space-fsnative_hemi-R.func.gii')

    par_l = surface.load_surf_data(par_l).T
    par_r = surface.load_surf_data(par_r).T

    par = np.concatenate((par_l, par_r))

    return cortex.Vertex(par, subject, alpha=alpha)


d = {}
for session in ['3t1', '7t1'][1:2]:
    for par in ['r2.grid', 'r2.optim', 'mu.grid', 'mu.optim', 'sd.grid', 'sd.optim']:
        for smoothed in [True]:
            for concatenated in [False, True]:
                key = f'{session}.{par}'

                if smoothed:
                    key += '.smoothed'
                if concatenated:
                    key += '.concatenated'
                         

                if par.startswith('mu') or par.startswith('sd'):
                    pass
                    values = _load_parameters(subject, session, par, smoothed,
                            concatenated=concatenated)
                            # alpha=d[f'{session}.r2.optim.smoothed'])
                    # values.data[d[f'{session}.r2.optim.smoothed.concatenated'].data < thr] = np.nan
                else:
                    values = _load_parameters(subject, session, par, smoothed,
                            concatenated=concatenated)
                # elif par.startswith('r2'):
                    # d[key + '.smoothed'] = cortex.Vertex(smooth(values.data), subject, factor=3.0, iterations=1)

                d[key] = values

ds = cortex.Dataset(**d)
cortex.webshow(d)
