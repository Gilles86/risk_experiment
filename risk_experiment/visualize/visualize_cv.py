import matplotlib.pyplot as plt
import os.path as op
from nilearn import surface
import cortex
import numpy as np
from risk_experiment.utils.argparse import run_main, make_default_parser
from utils import _load_parameters, get_alpha_vertex, get_wang15_ips

sourcedata = '/data/ds-risk'
subject = '02'
session = '3t1'
thr = .15
vmax = 28


def main(subject, session, sourcedata, standard_space=False, thr=thr, smoothed=False):
    left, right = cortex.db.get_surf(f'sub-{subject}', 'fiducial', merge=False)
    left, right = cortex.Surface(*left), cortex.Surface(*right)


    d = {}


    if standard_space:
        space = 'fsaverage'
        d['wang15_ips'] = get_wang15_ips('fsaverage', sourcedata)
        fs_subject = 'fsaverage'
    else:
        space = 'fsnative'
        d['wang15_ips'] = get_wang15_ips(subject, sourcedata)
        fs_subject = f'sub-{subject}'


    key = 'encoding_model.cv'

    if smoothed:
        key += '.smoothed'


    for session in ['3t2', '7t2']:

        r2_cv = []

        for run in range(1, 9):

            r2 = []

            for hemi in ['L', 'R']:
                r2.append(surface.load_surf_data(op.join(sourcedata, 'derivatives', f'{key}/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_run-{run}_desc-cvr2.optim_space-{space}_hemi-{hemi}.func.gii')))
            
            r2_cv.append(np.concatenate(r2))
        
        r2_cv = np.mean(np.clip(r2_cv, 0.0, np.inf), 0)
        print(np.quantile(r2_cv, .95))
        alpha = np.clip(r2_cv-thr, 0., .1) /.1
        d[f'r2_cv.{session}'] = get_alpha_vertex(r2_cv, alpha, cmap='hot',
                subject=subject, vmin=0.0, vmax=.5, standard_space=standard_space)


        r2 = _load_parameters(subject, session, 'r2.optim', space, smoothed)
        alpha = np.clip(r2.data-thr, 0., .1) /.1

        d[f'r2.{session}'] = get_alpha_vertex(r2.data, alpha, cmap='hot',
                subject=subject, vmin=0.0, vmax=.5, standard_space=standard_space)

    ds = cortex.Dataset(**d)
    cortex.webshow(ds)

    x = np.linspace(0, 1, 101, True)
    y = np.linspace(np.log(5), np.log(vmax), len(x), True)

    # plt.imshow(plt.cm.nipy_spectral(x)[np.newaxis, ...],
            # extent=[np.log(5),np.log(vmax), np.log(5), np.log(vmax)], aspect=1./10.,
            # origin='lower')

    # ns = np.array([5, 7, 10, 14, 20, 28, 40, 56, 80])
    # ns = ns[ns <= vmax]


    # plt.xticks(np.log(ns), ns)
    # plt.xlim(np.olg(5, np.log(vmax))
    # plt.show()

if __name__ == '__main__':
    parser = make_default_parser(sourcedata)
    parser.add_argument('--standard_space', action='store_true')
    parser.add_argument('--threshold', default=.15, type=float)
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()
    main(args.subject, args.session, args.bids_folder, standard_space=args.standard_space, thr=args.threshold,
            smoothed=args.smoothed)
