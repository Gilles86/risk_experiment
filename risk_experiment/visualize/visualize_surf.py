from tqdm.contrib.itertools import product
import matplotlib.pyplot as plt
import numpy as np
import cortex
from risk_experiment.utils.argparse import make_default_parser
from risk_experiment.utils import Subject, get_all_subject_ids
from utils import get_alpha_vertex
from scipy import stats as ss

vranges = {'mu':(5, 80), 'cvr2':(0.0, 0.2), 'r2':(0.0, 0.2)}
cmaps = {'mu':'nipy_spectral', 'cvr2':'afmhot', 'r2':'afmhot'}

def get_subject_vertices(subject, session, bids_folder, standard_space, thr, smoothed, show_unthresholded_map=False, show_pars=None):
    sub = Subject(subject, bids_folder)

    if standard_space:
        space = 'fsaverage'
        fs_subject = 'fsaverage'
    else:
        space = 'fsnative'
        fs_subject = f'sub-{subject}'

    pars = sub.get_prf_parameters_surf(session, space=space, smoothed=smoothed)

    thr1, thr2 = thr

    if session.endswith('2'):
        alpha = ss.norm(thr1, 0.025).cdf(pars['cvr2'].values)
    else:
        alpha = ss.norm(thr2, 0.025).cdf(pars['r2'].values)

    ds = {}

    if show_pars is None:
        keys = pars.columns
    else:
        keys = show_pars

    for key in keys:

        if show_unthresholded_map:
            ds[f'sub-{subject}.{key}'] = cortex.Vertex(pars[key].values, fs_subject)

        if key in vranges.keys():
            vmin, vmax = vranges[key]
            cmap = cmaps[key]
            ds[f'sub-{subject}.{session}.{key}_thr'] = get_alpha_vertex(pars[key].values, alpha, standard_space=standard_space, subject=fs_subject,
            vmin=vmin, vmax=vmax, cmap=cmap)
    
    return ds

def main(subject, session, bids_folder, standard_space=False, thr=-np.inf, smoothed=False, show_all_subjects=False, show_all_sessions=False, show_pars=None):

    print(f'***pars***: {show_pars}')

    if show_all_subjects:
        subjects = get_all_subject_ids()
    else:
        subjects = [subject]

    if show_all_sessions:
        sessions = ['3t1', '3t2', '7t1', '7t2']
    else:
        sessions = [session]

    ds = {}
    for subject, session in product(subjects, sessions):
        try:
            ds.update(get_subject_vertices(subject, session, bids_folder, standard_space, thr, smoothed, show_pars=show_pars))
        except Exception as e:
            print(f'Problem with subject/sessions {subject}/{session}: {e}')

    print(ds)
    ds = cortex.Dataset(**ds)

    cortex.webshow(ds)

    vmin, vmax = vranges['mu']
    x = np.linspace(0, 1, 101, True)
    im = plt.imshow(plt.cm.nipy_spectral(x)[np.newaxis, ...],
            extent=[vmin, vmax, 0, 1], aspect='auto',
            origin='lower')
    print(im.get_extent())

    ns = np.array([5, 7, 10, 14, 20, 28, 40, 56, 80])
    ns = ns[ns <= vmax]
    plt.xticks(ns, ns)
    print(ns)
    # plt.xlim(np.log(5), np.log(vmax))
    plt.show()

if __name__ == '__main__':
    parser = make_default_parser()
    parser.add_argument('--standard_space', action='store_true')
    parser.add_argument('--threshold', default=(-0.05, 0.01), type=float, nargs=2)
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--all_subjects', action='store_true')
    parser.add_argument('--all_sessions', action='store_true')
    parser.add_argument('--mean_subject', action='store_true')
    parser.add_argument('--pars', nargs='+', default=None)
    args = parser.parse_args()
    main(args.subject, args.session, args.bids_folder, standard_space=args.standard_space, thr=args.threshold,
            smoothed=args.smoothed, show_all_subjects=args.all_subjects, 
            show_pars=args.pars,
             show_all_sessions=args.all_sessions)
