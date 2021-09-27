import os.path as op
from nilearn import surface
import cortex
import numpy as np
from risk_experiment.utils.argparse import run_main, make_default_parser

sourcedata = '/data/ds-risk'
subject = '02'
session = '3t1'
thr = .15


def main(subject, session, sourcedata, standard_space=False, thr=thr):
    left, right = cortex.db.get_surf(f'sub-{subject}', 'fiducial', merge=False)
    left, right = cortex.Surface(*left), cortex.Surface(*right)


    def smooth(data, *args, **kwargs):
        n_left_pts = len(left.pts)

        smoothed_data = np.zeros_like(data)
        smoothed_data[:n_left_pts] = left.smooth(data[:n_left_pts], *args, **kwargs)
        smoothed_data[n_left_pts:] = right.smooth(data[n_left_pts:], *args, **kwargs)

        return smoothed_data


    def _load_parameters(subject, session, par, space='fsnative', smoothed=False, concatenated=False, alpha=None):


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

        kwargs = {}

        if par.startswith('r2'):
            kwargs['vmin'] = 0.0
            kwargs['vmax'] = 0.25

        if op.exists(par_l):

            par_l = surface.load_surf_data(par_l).T
            par_r = surface.load_surf_data(par_r).T

            par = np.concatenate((par_l, par_r))

            if space == 'fsnative':
                fs_subject = f'sub-{subject}'
            else:
                fs_subject = space

            

            return cortex.Vertex(par, fs_subject, alpha=alpha, **kwargs)
        else:
            return None


    d = {}

    if standard_space:
        space = 'fsaverage'
    else:
        space = 'fsnative'

    for session in ['3t1', '7t1'][:2]:
        for par in ['r2.grid', 'r2.optim', 'mu.grid', 'mu.optim', 'sd.grid', 'sd.optim', 'amplitude.optim']:
            for smoothed in [True]:
                for concatenated in [False]:
                    key = f'{session}.{par}'

                    if smoothed:
                        key += '.smoothed'
                    if concatenated:
                        key += '.concatenated'



                             

                    if par.startswith('mu') or par.startswith('sd'):
                        pass
                        values = _load_parameters(subject, session, par, 
                                space, smoothed,
                                concatenated=concatenated)
                        d[key] = values

                        values = _load_parameters(subject, session, par, 
                                space, smoothed,
                                concatenated=concatenated)
                        values.data[d[f'{session}.r2.optim.smoothed'].data < thr] = np.nan
                        key += '.thr'
                        
                        if par[:2] == 'mu':
                            values = _load_parameters(subject, session, par,
                                    space, smoothed,
                                    concatenated=concatenated)
                            values.vmin = 1
                            values.vmax = 28
                            values.data = np.exp(values.data)
                            values.data[d[f'{session}.r2.optim.smoothed'].data < thr] = np.nan
                            key += '.natural'

                    else:
                        values = _load_parameters(subject, session, par,
                                space, smoothed,
                                concatenated=concatenated)

                    if values:
                        d[key] = values

    ds = cortex.Dataset(**d)
    cortex.webshow(ds)

if __name__ == '__main__':
    parser = make_default_parser(sourcedata)
    parser.add_argument('--standard_space', action='store_true')
    parser.add_argument('--threshold', default=.15, type=float)
    args = parser.parse_args()
    main(args.subject, args.session, args.bids_folder, standard_space=args.standard_space, thr=args.threshold)
