import argparse
import pandas as pd
from tqdm.contrib.itertools import product
import matplotlib.pyplot as plt
import numpy as np
import cortex
from risk_experiment.utils.argparse import make_default_parser
from risk_experiment.utils import get_all_subjects
from utils import get_alpha_vertex
from scipy import stats as ss

vranges = {'mu':(5, 80), 'cvr2':(0.0, 0.025), 'r2':(0.0, 0.2), 'mu_natural':(5, 80), 'sd':(2, 12)}
cmaps = {'mu':'nipy_spectral', 'cvr2':'afmhot', 'r2':'afmhot', 'mu_natural':'nipy_spectral', 
'sd':'hot'}

def get_average_vertex(session, bids_folder, thr, smoothed, show_unthresholded_map=False, show_pars=None,
threshold_on='r2'):


    subjects = get_all_subjects(bids_folder)

    pars = []

    for sub in subjects:
        try:
            pars.append(sub.get_prf_parameters_surf(session, space='fsaverage', smoothed=smoothed))
        except Exception as e:
            print(f'Problem with subject {sub.subject}: {e} ')
    
    pars = pd.concat(pars, keys=[sub.subject for sub in subjects], names=['subject'])
    # pars['mu_natural'] = np.exp(pars['mu'])

    if 'cvr2' in pars.columns:
        pars['cvr2'] = np.clip(pars['cvr2'], -0.01, 1)
        pars.loc[pars['r2'] < 0.01, 'r2'] = 0.0
        print(pars['cvr2'])
        print(pars['r2'])

    mean_pars = pars.groupby(['hemi', 'vertex']).mean()

    alpha = np.round(ss.norm(thr, 0.005).cdf(mean_pars[threshold_on].values), 2)


    ds = {}

    if show_pars is None:
        keys = pars.columns
    else:
        keys = show_pars

    print(keys)
    for key in keys:

        if show_unthresholded_map:
            ds[f'{session}.{key}'] = cortex.Vertex(pars[key].values, 'fsaverage')

        if key in vranges.keys():
            vmin, vmax = vranges[key]
            cmap = cmaps[key]
            ds[f'{session}.{key}_thr'] = get_alpha_vertex(mean_pars[key].values, alpha, standard_space=True, subject='fsaverage',
            vmin=vmin, vmax=vmax, cmap=cmap)
    
    return ds

def main(sessions, bids_folder, thr=-np.inf, smoothed=False, show_pars=None):

    if sessions is None:
        # sessions = ['3t2', '7t2']
        sessions = ['3t1', '3t2', '7t1', '7t2']

    thresholds = {'3t1':0.065, '3t2':0.04, '7t1':0.065, '7t2':0.04}
    ds = {}
    # for session in sessions:
    #     ds.update(get_average_vertex(session, bids_folder, thresholds[session], smoothed, show_pars=show_pars))

    # ds = cortex.Dataset(**ds)

    # cortex.webshow(ds)

    vmin, vmax = vranges['mu']
    x = np.linspace(0, 1, 101, True)
    
    # Width is 80 x 1, so 
    im = plt.imshow(plt.cm.nipy_spectral(x)[np.newaxis, ...],
            extent=[vmin, vmax, 0, 1], aspect=1.*(vmax-vmin) / 20.,
            origin='lower')
    print(im.get_extent())
    plt.yticks([])
    plt.tight_layout()

    ns = np.array([5, 7, 10, 14, 20, 28, 40, 56, 80])
    ns = ns[ns <= vmax]
    plt.xticks(ns)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pars', nargs='+', default=None)
    parser.add_argument('--sessions', nargs='+', default=None)
    args = parser.parse_args()
    main(args.sessions, args.bids_folder, thr=args.threshold,
            smoothed=args.smoothed,)
