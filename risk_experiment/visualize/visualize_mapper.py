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
    else:
        space = 'fsnative'
        d['wang15_ips'] = get_wang15_ips(subject, sourcedata)


    concatenated = False

    for session in ['3t1', '7t1', '3t2', '7t2']:

        # for desc, split_certainty in zip(['', '.certain', '.uncertain'], [False, True, True]):
        for desc, split_certainty in zip([''], [False]):
            ext = ''
            # if pca_confounds:
                # ext += '.pca_confounds'
            if smoothed:
                ext += '.smoothed'


            print(desc)
            r2 = _load_parameters(subject, session, 'r2.optim'+desc, space, smoothed,
                    split_certainty=split_certainty,
                    concatenated=concatenated, pca_confounds=False)

            if r2 is not None:
                alpha = np.clip(r2.data-thr, 0., .1) /.1

                d[f'r2.{session}{ext}{desc}'] = get_alpha_vertex(r2.data, alpha, cmap='hot',
                        subject=subject, vmin=0.0, vmax=.5,
                        standard_space=standard_space)

                mu = _load_parameters(subject, session, 'mu.optim'+desc, space, smoothed,
                        split_certainty=split_certainty,
                        concatenated=concatenated)

                d[f'mu.{session}{ext}{desc}'] = get_alpha_vertex(mu.data, alpha, cmap='nipy_spectral',
                        subject=subject, vmin=np.log(5), vmax=np.log(28),
                        standard_space=standard_space)



    ds = cortex.Dataset(**d)
    cortex.webshow(ds)

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

if __name__ == '__main__':
    parser = make_default_parser(sourcedata)
    parser.add_argument('--standard_space', action='store_true')
    parser.add_argument('--threshold', default=.15, type=float)
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()
    main(args.subject, args.session, args.bids_folder, standard_space=args.standard_space, thr=args.threshold,
            smoothed=args.smoothed)
