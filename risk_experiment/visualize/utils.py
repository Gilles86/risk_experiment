import cortex
from nilearn import surface
import numpy as np
import os.path as op
from matplotlib import colors, cm

def get_alpha_vertex(data, alpha, cmap='nipy_spectral', vmin=np.log(5), vmax=np.log(80), standard_space=False, subject='fsaverage'):

    data = np.clip((data - vmin) / (vmax - vmin), 0., .99)
    red, green, blue = getattr(cm, cmap)(data,)[:, :3].T

    if (subject == 'fsaverage') or standard_space:
        fs_subject = 'fsaverage'
    else:
        fs_subject = f'sub-{subject}'

    print(fs_subject)
    # Get curvature
    curv = cortex.db.get_surfinfo(fs_subject)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map.
    curv.data = np.sign(curv.data.data) * .25
    curv.vmin = -1
    curv.vmax = 1
    curv.cmap = 'gray'
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data]).astype(np.float32)

    vx_rgb = (np.vstack([red.data, green.data, blue.data]) * 255.).astype(np.float32)

    display_data = vx_rgb * alpha[np.newaxis, :] + curv_rgb * (1.-alpha[np.newaxis, :])

    return cortex.VertexRGB(*display_data.astype(np.uint8), fs_subject)

def _load_parameters(subject, session, par, space='fsnative', smoothed=False, concatenated=False,
        pca_confounds=False, alpha=None, split_certainty=False,
        sourcedata='/data/ds-risk', **kwargs):


    d_name = 'encoding_model'
    if smoothed:
        d_name += '.smoothed'
    if pca_confounds:
        d_name += '.pca_confounds'
    if concatenated:
        d_name += '.concatenated'

    if split_certainty:
        d_name += '.split_certainty'

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

         

        if alpha is None:
            return cortex.Vertex(par, fs_subject, **kwargs)
        else:
            get_alpha_vertex(par, alpha, subject=fs_subject, **kwargs)

    else:
        return None


def get_wang2015(subject, sourcedata, probabilistic=False):

        if subject == 'fsaverage':
            fs_subject = 'fsaverage'
            w2015_l = surface.load_surf_data(op.join(sourcedata, 'derivatives', 'freesurfer', fs_subject,
                'surf', 'lh.wang2015_atlas.mgz'))
            w2015_r = surface.load_surf_data(op.join(sourcedata, 'derivatives', 'freesurfer', fs_subject,
                'surf', 'rh.wang2015_atlas.mgz'))
        else:
            fs_subject = f'sub-{subject}'

            if probabilistic:
                w2015_l = surface.load_surf_data(op.join(sourcedata, 'derivatives', 'freesurfer', fs_subject,
                    'surf', 'lh.wang15_fplbl.mgz'))
                w2015_r = surface.load_surf_data(op.join(sourcedata, 'derivatives', 'freesurfer', fs_subject,
                    'surf', 'rh.wang15_fplbl.mgz'))
            else:
                w2015_l = surface.load_surf_data(op.join(sourcedata, 'derivatives', 'freesurfer', fs_subject,
                    'surf', 'lh.wang15_mplbl.mgz'))
                w2015_r = surface.load_surf_data(op.join(sourcedata, 'derivatives', 'freesurfer', fs_subject,
                    'surf', 'rh.wang15_mplbl.mgz'))

        if probabilistic:
            w2015 = np.concatenate((w2015_l, w2015_r), 1)
        else:
            w2015 = np.concatenate((w2015_l, w2015_r), 0)

        return cortex.Vertex(w2015, fs_subject)


def get_wang15_ips(subject, sourcedata):

    if subject == 'fsaverage':
        fs_subject = subject
    else:
        fs_subject = f'sub-{subject}'
    
    fn = op.join(sourcedata, 'derivatives', 'freesurfer', fs_subject, 'surf',
            '{hemi}.wang15_ips.mgz')
    surfs = [surface.load_surf_data(fn.format(hemi=hemi)) for hemi in ['lh', 'rh']]

    surfs = np.concatenate(surfs)
    return cortex.Vertex(surfs, fs_subject)

