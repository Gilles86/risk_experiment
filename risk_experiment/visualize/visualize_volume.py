import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
import os.path as op
from nilearn import surface, image
import cortex
import numpy as np
from risk_experiment.utils.argparse import run_main, make_default_parser
from utils import _load_parameters, get_alpha_vertex

cmap = 'nipy_spectral'

sourcedata = '/data/ds-risk'
subject = '10'
session = '7t1'
thr = .15

d = {}

for session in ['3t1', '3t2', '7t1', '7t2']:
    if session.endswith('1'):
        threshold = 0.055
    else:
        threshold = 0.03

    d[f'r2_{session}_vol'] = _load_parameters(subject, session, 'r2', volume=True, vmin=0.0, vmax=.5,
            cmap='hot', threshold=threshold)
    d[f'mu_{session}_vol'] =_load_parameters(subject, session, 'mu', volume=True, vmin=np.log(5), vmax=np.log(28),
            alpha=d[f'r2_{session}_vol'].alpha,
            cmap='nipy_spectral')


    # d[f'r2_{session}_surf'] = _load_parameters(subject, session, 'r2', vmin=0.0, vmax=.5,
            # cmap='hot', threshold=threshold)
    # mu = _load_parameters(subject, session, 'mu', 
            # cmap='hot', threshold=threshold)

    # alpha = np.clip(d[f'r2_{session}_surf'].data-threshold, 0., .1) /.1
    # d[f'mu_{session}_surf'] = get_alpha_vertex(mu.data, alpha, cmap='nipy_spectral',
            # subject=subject, vmin=np.log(5), vmax=np.log(28),
            # standard_space=False)

    d[f'r2_{session}_volsurf'] = _load_parameters(subject, session, 'r2.volume', vmin=0.0, vmax=.5,
            cmap='hot', threshold=threshold)
    mu = _load_parameters(subject, session, 'mu.volume', 
            cmap='hot', threshold=threshold)

    alpha = np.clip(d[f'r2_{session}_volsurf'].data-threshold, 0., .1) /.1
    d[f'mu_{session}_volsurf'] = get_alpha_vertex(mu.data, alpha, cmap='nipy_spectral',
            subject=subject, vmin=np.log(5), vmax=np.log(28),
            standard_space=False)


print(d)
ds = cortex.Dataset(**d)

cortex.webshow(ds)

vmax = 28
x = np.linspace(0, 1, 101, True)
y = np.linspace(np.log(5), np.log(vmax), len(x), True)

plt.imshow(plt.cm.nipy_spectral(x)[np.newaxis, ...],
        extent=[np.log(5),np.log(vmax), np.log(5), np.log(vmax)], aspect=1./10.,
        origin='lower')

ns = np.array([5, 7, 10, 14, 20, 28, 40, 56, 80])
ns = ns[ns <= vmax]


plt.xticks(np.log(ns), ns)
# plt.xlim(np.olg(5, np.log(vmax))
plt.show()
# pc_subject = f'sub-{subject}'

# parameter = 'r2'
# r2 = image.load_img(op.join(sourcedata, 'derivatives', 'encoding_model', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_desc-{parameter}.optim_space-T1w_pars.nii.gz'))


# parameter = 'mu'
# vol_mu = op.join(sourcedata, 'derivatives', 'encoding_model', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_desc-{parameter}.optim_space-T1w_pars.nii.gz')


# bold = image.load_img(op.join(sourcedata, 'derivatives', 'fmriprep', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-mapper_run-1_space-T1w_boldref.nii.gz'))

# transform = cortex.xfm.Transform(np.identity(4), bold)
# transform.save(pc_subject, f'bold.{session}')

# vmin, vmax = 1, 4.
# data = image.load_img(vol_mu).get_data()
# data = np.clip((data - vmin) / (vmax - vmin), 0., .99)
# rgb = getattr(cm, cmap)(data.T,)[..., :3]
# red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]

# alpha = np.clip((r2.get_data().T - .05) / .05 + .2, 0.0, 1.0)

# # cortex.webshow(cortex.VolumeRGB(red, green, blue, pc_subject, f'bold.{session}'))# alpha=alpha))
# cortex.webshow(cortex.VolumeRGB(red, green, blue, pc_subject, f'bold.{session}', alpha=alpha))

# # r2_vol = cortex.Volume(vol_data, pc_subject, f'bold.{session}', vmin=0, vmax=.5)
# # vol_mu = cortex.Volume(vol_mu, pc_subject, f'bold.{session}', vmin=0, vmax=.5,
        # # alpha=image.new_img_like(vol_mu, np.clip(r2_vol.data - .1 / .2, 0, 1.)))

# # r2_surf = _load_parameters(subject, session, 'r2.optim', sourcedata)

# # bold_volume = cortex.Volume(bold, pc_subject, f'bold.{session}', vmin=0, vmax=.5,
        # # alpha=vol_data)


# # ds = cortex.Dataset(r2=r2_vol, vol_mu=vol_mu)
# # cortex.webshow(ds)



