import os.path as op
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn import surface
import nibabel as nb
import pandas as pd
import numpy as np
from nibabel import gifti

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def get_sourcedata():
    return '/data/ds-risk'

def get_runs(subject, session):

    if isinstance(subject, int):
        subject = f'{subject:02d}'

    if (subject == '08') & (session == '7t1'):
        return [1, 2, 4, 5]

    if session[-1] == '1':
        return range(1, 5)
    else:
        return range(1, 9)

def get_behavior(subject, session, sourcedata):

    runs = get_runs(subject, session)

    behavior = []

    for run in runs:
        behavior.append(pd.read_table(op.join(
            sourcedata, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_events.tsv')))

    behavior = pd.concat(behavior, keys=runs, names=['run'])
    behavior['subject'] = int(subject)
    behavior = behavior.reset_index().set_index(
        ['subject', 'run', 'trial_type'])

    return behavior


def get_fmriprep_confounds(subject, session, sourcedata,
                           confounds_to_include=None):

    print(f'Getting fmriprepconfounds for {subject} - {session}')

    runs = get_runs(subject, session)

    print(runs)

    if confounds_to_include is None:
        fmriprep_confounds_include = ['dvars', 'framewise_displacement', 'trans_x',
                                      'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                      'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
                                      'cosine00', 'cosine01', 'cosine02',
                                      'non_steady_state_outlier00', 'non_steady_state_outlier01',
                                      'non_steady_state_outlier02']
    else:
        fmriprep_confounds_include = confounds_to_include


    fmriprep_confounds = [
            op.join(sourcedata, f'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_desc-confounds_timeseries.tsv') for run in runs]
    fmriprep_confounds = [pd.read_table(
        cf)[fmriprep_confounds_include] for cf in fmriprep_confounds]

    fmriprep_confounds = [rc.set_index(get_frametimes(subject, session)) for rc in fmriprep_confounds]

    confounds =  pd.concat(fmriprep_confounds, 0, keys=runs, names=['run'])
    return confounds.groupby('run').transform(lambda x: x.fillna(x.mean()))


def get_retroicor_confounds(subject, session, sourcedata):
    runs = get_runs(subject, session)
    retroicor_confounds = [
            op.join(sourcedata, f'derivatives/physiotoolbox/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_desc-retroicor_timeseries.tsv') for run in runs]
    retroicor_confounds = [pd.read_table(
        cf, header=None, usecols=range(18)) for cf in retroicor_confounds]

    n_vols = len(retroicor_confounds[0])
    tr = get_tr(subject, session)

    retroicor_confounds = [rc.set_index(get_frametimes(subject, session)) for rc in retroicor_confounds]

    confounds =  pd.concat(retroicor_confounds, 0, keys=runs, names=['run'])
    confounds.columns = [f'retroicor_{i:02d}' for i in range(confounds.shape[1])]
    return confounds.groupby('run').transform(lambda x: x.fillna(x.mean()))


def get_confounds(subject, session,  sourcedata=None):

    if sourcedata is None:
        sourcedata = get_sourcedata()


def get_tr(subject, sesion):
    return 2.3

def get_frametimes(subject, session):
    if session[-1] == '1':
        n_vols = 125
    else:
        raise NotImplementedError

    tr = get_tr(subject, session)
    return np.linspace(0, (n_vols-1)*tr, n_vols)


def get_mapper_response_hrf(subject, session, sourcedata):

    assert(session[-1] == '1')

    behavior = get_behavior(subject, session, sourcedata)

    responses = behavior.xs('response', 0, 'trial_type', drop_level=False).reset_index(
        'trial_type')[['onset', 'trial_type']]
    responses['duration'] = 0.0
    responses = responses[responses.onset > 0]

    tr = get_tr(subject, session)
    frametimes = np.linspace(0, (125-1)*tr, 125)

    response_hrf = responses.groupby('run').apply(lambda d: make_first_level_design_matrix(frametimes,
                                                                                          d, drift_model=None,
                                                                                          drift_order=0))
    
    return response_hrf[['response']]

def get_surf_file(subject, session, run, sourcedata,
        hemi='lh', space='fsnative'):

    if session[-1] == '1':
        task = 'mapper'
    elif session[-1] == '2':
        task = 'task'


    if hemi == 'lh':
        hemi = 'L'
    elif hemi == 'rh':
        hemi = 'R'

    dir_ = op.join(sourcedata, 'derivatives', 'fmriprep', f'sub-{subject}',
            f'ses-{session}', 'func')

    fn = op.join(dir_, f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')

    return fn


def get_surf_data(subject, session, sourcedata, smoothed=False, space='fsnative'):

    runs  = get_runs(subject, session)

    if smoothed:
        dir_ = op.join(sourcedata, 'derivatives', 'smoothed', f'sub-{subject}',
                f'ses-{session}', 'func',)
    else:
        dir_ = op.join(sourcedata, 'derivatives', 'fmriprep', f'sub-{subject}',
                f'ses-{session}', 'func',)

    frametimes = get_frametimes(subject, session)

    data = []
    for run in runs:
        d_ = []
        for hemi in ['L', 'R']:
            if smoothed:
                d =  op.join(dir_,
                        f'sub-{subject}_ses-{session}_task-mapper_run-{run}_space-{space}_hemi-{hemi}_desc-smoothed_bold.func.gii')
            else:
                d =  op.join(dir_,
                        f'sub-{subject}_ses-{session}_task-mapper_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')
            d = surface.load_surf_data(d).T
            columns = pd.MultiIndex.from_product([[hemi], np.arange(d.shape[1])], names=['hemi', 'vertex'])
            index = pd.MultiIndex.from_product([[run], frametimes], names=['run', None])
            d_.append(pd.DataFrame(d, columns=columns, index=index))

        data.append(pd.concat(d_, 1))
    return pd.concat(data, 0)


def get_mapper_paradigm(subject, session, sourcedata):

    run = 1
    events = pd.read_table(op.join(
            sourcedata, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_events.tsv'))

    events = events[events['trial_type'] == 'stimulation'].sort_values('onset')
    events['onset_halfway'] = events['onset']+events['duration'] / 2.
    events.index = pd.to_timedelta(events.onset_halfway, unit='s')
    tmp = pd.DataFrame([0,0], columns=['n_dots'])
    n_vols = 125
    tr = get_tr(subject, session)
    tmp.index = pd.to_timedelta([0, (n_vols-1)*tr], 's')
    paradigm = pd.concat((tmp, events)).n_dots.resample('2.3S').nearest().to_frame('n_dots').astype(np.float32)
    paradigm['n_dots'] = np.log(paradigm['n_dots']).replace(-np.inf, 0)
    
    return paradigm


def get_target_dir(subject, session, sourcedata, base, modality='func'):
    target_dir =  op.join(sourcedata, 'derivatives', base, f'sub-{subject}', f'ses-{session}',
            modality)

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    return target_dir

def write_gifti(subject, session, sourcedata, space, data, filename):
    dir_ = op.join(sourcedata, 'derivatives', 'fmriprep', f'sub-{subject}',
            f'ses-{session}', 'func',)

    run = 1

    if data.ndim == 1:
        data = data.to_frame().T

    for hemi, d in data.groupby(['hemi'], axis=1):
        header = nb.load(op.join(dir_,
            f'sub-{subject}_ses-{session}_task-mapper_run-{run}_space-{space}_hemi-L_bold.func.gii')).header
        print(d)
        darrays = [nb.gifti.GiftiDataArray(data=d_.values) for _, d_ in d.iterrows()]
        im = gifti.GiftiImage(header=header,
                darrays=darrays)
        im.to_filename(filename.format(hemi=hemi))
