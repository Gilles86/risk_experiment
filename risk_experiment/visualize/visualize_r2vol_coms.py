import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
import os.path as op
from nilearn import surface, image
import cortex
import numpy as np
from risk_experiment.utils.argparse import run_main, make_default_parser
from utils import _load_parameters, get_alpha_vertex
from tqdm.contrib.itertools import product


bids_folder = '/data/ds-risk'

# subjects = ['02', '03', '04', '05']
subjects = [f'{x:02d}' for x in range(2, 32)]


problems = ['03', '08', '10', '14', '18', '23', '29']
for p in problems:
    subjects.pop(subjects.index(p))

subjects.pop(subjects.index('24'))
smoothed = False
sessions = ['3t2']

d = {}

coms = []

for subject, session in product(subjects, sessions):
    if session.endswith('1'):
        threshold = 0.025
    else:
        threshold = 0.02

    # if (session == '3t2') & (int(subject) < 11):
        # euclidean_distance = np.concatenate([surface.load_surf_data(op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subject}',
            # 'surf', f'{hemi}.npc_euclideandistance_space-fsaverage.mgz')) for hemi in ['lh', 'rh']])

        # d[f'euclidean_distance_{subject}'] = cortex.Vertex(euclidean_distance, subject=f'fsaverage', vmin=0.1, vmax=50.0)

        # geodesic_distance = np.concatenate([surface.load_surf_data(op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subject}',
            # 'surf', f'{hemi}.npc_geodesicdistance_space-fsaverage.mgz')) for hemi in ['lh', 'rh']])

        # d[f'geodesic_distance_{subject}'] = cortex.Vertex(geodesic_distance, subject=f'fsaverage', vmin=0.1, vmax=50.0)

    r2 = _load_parameters(subject, session, 'r2.volume', vmin=0.0, vmax=.5,
            cmap='hot', threshold=threshold, space='fsaverage',
            smoothed=smoothed)

    threshold = np.nanpercentile(r2.data, 75)

    alpha = np.clip(r2.data-threshold, 0., threshold*2.) /threshold/2.


    mu = _load_parameters(subject, session, 'mu.volume', 
            cmap='hot', threshold=threshold, space='fsaverage',
            smoothed=smoothed)

    d[f'r2_{subject}.{session}_volsurf'] = get_alpha_vertex(r2.data, alpha, cmap='hot',
            subject=subject, vmin=0.0, vmax=0.4,
            standard_space=True)

    # d[f'mu_{subject}.{session}_volsurf'] = get_alpha_vertex(mu.data, alpha, cmap='nipy_spectral',
            # subject=subject, vmin=np.log(5), vmax=np.log(28),
            # standard_space=True)

    # com = surface.load_surf_data(op.join(bids_folder, 'derivatives', 'npc_com', f'sub-{subject}_space-fsaverage_desc-npcr_com.gii'))

    # com = np.concatenate((np.zeros_like(com), com))

    com = surface.load_surf_data(op.join(bids_folder, 'derivatives', 'tms_targets', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_desc-r2com.volume.optim_space-fsaverage_hemi-R.func.gii'))
    com = np.concatenate((np.zeros_like(com), com))

    coms.append(com)

    d[f'com_r2vol_npcr_{subject}.{session}_volsurf'] = cortex.Vertex(com, 'fsaverage', vmin=.5, vmax=1.0)

all_com = np.mean(coms, 0)
# all_com[all_com == 0.0] = np.nan

# d['all_coms'] = cortex.Vertex(all_com, 'fsaverage', alpha=(all_com !=0.0).astype(float))
d['all_coms'] = get_alpha_vertex(all_com, (all_com !=0.0).astype(float), cmap='hot',
            subject=subject, vmin=0.0, vmax=40,
            standard_space=True)


ds = cortex.Dataset(**d)

cortex.webshow(ds)
