import os.path as op
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn import surface, image
from nilearn.input_data import NiftiMasker
import nibabel as nb
import pandas as pd
import numpy as np
from nibabel import gifti
from tqdm import tqdm
from tqdm.contrib.itertools import product
from sklearn.decomposition import PCA

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def get_sourcedata():
    return '/data/ds-risk'


def get_all_sessions():
    return ['3t1', '3t2', '7t1', '7t2']


def get_runs(subject, session):

    if isinstance(subject, int):
        subject = f'{subject:02d}'

    if (subject == '08') & (session == '7t1'):
        return [1, 2, 4, 5]

    if (subject == '23') & (session == '7t1'):
        return [1, 2, 3, 5]

    if session[-1] == '1':
        return range(1, 5)
    else:
        return range(1, 9)

def get_all_subject_ids():
    subjects = ['%02d' % i for i in range(2, 33)]
    subjects.pop(subjects.index('24'))
    return subjects

def get_all_subjects(bids_folder):
    return [Subject(subject, bids_folder) for subject in get_all_subject_ids()]

def get_all_behavior(sessions=['3t2', '7t2'], bids_folder='/data'):
    subjects = get_all_subjects(bids_folder=bids_folder)
    behavior = [s.get_behavior(sessions=sessions) for s in tqdm(subjects)]
    return pd.concat(behavior)

class Subject(object):

    def __init__(self, subject, bids_folder='/data'):

        self.subject = '%02d' % int(subject)
        self.bids_folder = bids_folder

    def get_volume_mask(self, roi='NPC12r'):

        if roi.startswith('NPC'):
            return op.join(self.derivatives_dir
            ,'ips_masks',
            f'sub-{self.subject}',
            'anat',
            f'sub-{self.subject}_space-T1w_desc-{roi}_mask.nii.gz'
            )

        else:
            raise NotImplementedError

    @property
    def derivatives_dir(self):
        return op.join(self.bids_folder, 'derivatives')

    @property
    def fmriprep_dir(self):
        return op.join(self.derivatives_dir, 'fmriprep', f'sub-{self.subject}')

    @property
    def t1w(self):
        t1w = op.join(self.fmriprep_dir,
        'anat',
        'sub-{self.subject}_desc-preproc_T1w.nii.gz')

        if not op.exists(t1w):
            raise Exception(f'T1w can not be found for subject {self.subject}')

        return t1w

    def get_nprf_pars(self, session=1, model='encoding_model.smoothed', parameter='r2',
    volume=True):

        if not volume:
            raise NotImplementedError

        im = op.join(self.derivatives_dir, model, f'sub-{self.subject}',
        f'ses-{session}', 'func', 
        f'sub-{self.subject}_ses-{session}_desc-{parameter}.optim_space-T1w_pars.nii.gz')

        return im

    def get_behavior(self, sessions=None):

        if sessions is None:
            sessions = ['3t2', '7t2']

        if type(sessions) is not list:
            sessions = [sessions]

        df = []
        for session in sessions:

            runs = get_runs(self.subject, session)
            for run in runs:

                fn = op.join(self.bids_folder, f'sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-task_run-{run}_events.tsv')

                if op.exists(fn):
                    d = pd.read_csv(fn, sep='\t',
                                index_col=['trial_nr', 'trial_type'])
                    d['subject'], d['session'], d['run'] = int(self.subject), session, run
                    df.append(d)

        if len(df) > 0:
            df = pd.concat(df)
            df = df.reset_index().set_index(['subject', 'session', 'run', 'trial_nr', 'trial_type']) 
            df = df.unstack('trial_type')
            return self._cleanup_behavior(df)
        else:
            return pd.DataFrame([])

    @staticmethod
    def _cleanup_behavior(df_):
        df = df_[[]].copy()
        df['rt'] = df_.loc[:, ('onset', 'choice')] - df_.loc[:, ('onset', 'stimulus 2')]
        df['certainty'] = df_.loc[:, ('choice', 'certainty')]
        df['n1'], df['n2'] = df_['n1']['stimulus 1'], df_['n2']['stimulus 1']
        df['prob1'], df['prob2'] = df_['prob1']['stimulus 1'], df_['prob2']['stimulus 1']

        df['choice'] = df_[('choice', 'choice')]
        df['risky_first'] = df['prob1'] == 0.55
        df['chose_risky'] = (df['risky_first'] & (df['choice'] == 1.0)) | (~df['risky_first'] & (df['choice'] == 2.0))
        df.loc[df.choice.isnull(), 'chose_risky'] = np.nan


        df['n_risky'] = df['n1'].where(df['risky_first'], df['n2'])
        df['n_safe'] = df['n2'].where(df['risky_first'], df['n1'])
        df['frac'] = df['n_risky'] / df['n_safe']
        df['log(risky/safe)'] = np.log(df['frac'])

        def get_risk_bin(d):
            try: 
                return pd.qcut(d, 6, range(1, 7))
            except Exception as e:
                n = len(d)
                ix = np.linspace(1, 7, n, False)

                d[d.sort_values().index] = np.floor(ix)
                
                return d
        df['bin(risky/safe)'] = df.groupby(['subject'])['frac'].apply(get_risk_bin)


        df = df[~df.chose_risky.isnull()]
        df['chose_risky'] = df['chose_risky'].astype(bool)
        return df.droplevel(-1, 1)
        

def get_fmriprep_confounds(subject, session, sourcedata,
                           confounds_to_include=None):


    runs = get_runs(subject, session)

    if session.endswith('1'):
        task = 'mapper'
    elif session.endswith('2'):
        task = 'task'
    else:
        raise ValueError(f'Invalid session: {session}')


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
        op.join(sourcedata, f'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-{task}_run-{run}_desc-confounds_timeseries.tsv') for run in runs]
    fmriprep_confounds = [pd.read_table(
        cf)[fmriprep_confounds_include] for cf in fmriprep_confounds]

    fmriprep_confounds = [rc.set_index(get_frametimes(
        subject, session, run)) for run, rc in zip(runs, fmriprep_confounds)]

    # confounds = pd.concat(fmriprep_confounds, 0, keys=runs, names=['run'])
    return fmriprep_confounds


def get_retroicor_confounds(subject, session, sourcedata, n_cardiac=2, n_respiratory=2, n_interaction=0):

    runs = get_runs(subject, session)

    if ((subject == '25') & (session == '7t1')) | ((subject == '10') & (session == '3t2')):
        print('No physiological data')
        index = pd.MultiIndex.from_product(
            [runs, get_frametimes(subject, session)], names=['run', None])
        return pd.DataFrame(index=index, columns=[])

    if session.endswith('1'):
        task = 'mapper'
        nvols = 125
    elif session.endswith('2'):
        task = 'task'
        nvols = 160
    else:
        raise ValueError(f'Invalid session: {session}')

    # Make columns
    columns = []
    for n, modality in zip([3, 4, 2], ['cardiac', 'respiratory', 'interaction']):
        for order in range(1, n+1):
            columns += [(modality, order, 'sin'), (modality, order, 'cos')]

    columns = pd.MultiIndex.from_tuples(
        columns, names=['modality', 'order', 'type'])

    retroicor_confounds = [
        op.join(sourcedata, f'derivatives/physiotoolbox/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-{task}_run-{run}_desc-retroicor_timeseries.tsv') for run in runs]
    retroicor_confounds = [pd.read_table(
        cf, header=None, usecols=range(18), names=columns) if op.exists(cf) else pd.DataFrame(np.zeros((nvols,0))) for cf in retroicor_confounds ]

    retroicor_confounds = [rc.set_index(get_frametimes(
        subject, session, run)) for run, rc in zip(runs, retroicor_confounds)]

    confounds = pd.concat(retroicor_confounds, 0, keys=runs,
                          names=['run']).sort_index(axis=1)

    confounds = pd.concat((confounds.loc[:, ('cardiac', slice(n_cardiac))],
                           confounds.loc[:, ('respiratory',
                                             slice(n_respiratory))],
                           confounds .loc[:, ('interaction', slice(n_interaction))]), axis=1)

    confounds = [cf.droplevel('run') for _, cf in confounds.groupby(['run'])]
    return confounds

def get_confounds(subject, session, bids_folder, include_fmriprep=None, pca=False, pca_n_components=.95):
    
    fmriprep_confounds = get_fmriprep_confounds(subject, session, bids_folder, confounds_to_include=include_fmriprep)
    retroicor_confounds = get_retroicor_confounds(subject, session, bids_folder, n_cardiac=2, n_respiratory=2, n_interaction=0)
    confounds = [pd.concat((rcf, fcf), axis=1) for rcf, fcf in zip(retroicor_confounds, fmriprep_confounds)]
    confounds = [c.fillna(method='bfill') for c in confounds]

    original_size = confounds[0].shape[1]

    if pca:
        def map_cf(cf, n_components=pca_n_components):
            pca = PCA(n_components=n_components)
            cf -= cf.mean(0)
            cf /= cf.std(0)
            cf = pd.DataFrame(pca.fit_transform(cf))
            cf.columns = [f'pca_{i}' for i in range(1, cf.shape[1]+1)]
            return cf
        confounds = [map_cf(cf) for cf in confounds]

    new_size = np.mean([cf.shape[1] for cf in confounds])
    print(f"RESIZED CONFOUNDS: {original_size} to {new_size}")

    return confounds

def get_tr(subject, sesion):
    return 2.3


def get_frametimes(subject, session, run=None):

    if session[-1] == '1':
        n_vols = 125
    else:
        n_vols = 160

    if (session == '7t2') & (subject == '02') & (run == 1):
        n_vols = 213

    

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

    fn = op.join(
        dir_, f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')

    return fn


def get_single_trial_surf_data(subject, session, bids_folder, smoothed=False,
                          space='fsnative', hemi=None, mask=None):

    if hemi is None:
        d_l = get_single_trial_surf_data(subject, session, bids_folder,
                                    smoothed, space, hemi='L', mask=mask)
        d_r = get_single_trial_surf_data(subject, session, bids_folder,
                                    smoothed, space, hemi='R', mask=mask)

        d = pd.concat((d_l, d_r), axis=1, keys=['L', 'R'])

        return d
    else:

        key = 'glm_stim1_surf'

        if smoothed:
            key += '.smoothed'

        data = surface.load_surf_data(op.join(bids_folder, 'derivatives', key,
                                              f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-task_space-fsnative_desc-stims1_hemi-{hemi}.pe.gii')).T
        data= pd.DataFrame(data, columns=pd.Index(np.arange(data.shape[1]), name='vertex'))

        if mask is not None:
            mask = get_surf_mask(subject, mask, hemi, bids_folder)
            data = data.loc[:, mask.values]
        return data



def get_surf_data(subject, session, sourcedata, smoothed=False, space='fsnative'):

    runs = get_runs(subject, session)

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
                d = op.join(dir_,
                            f'sub-{subject}_ses-{session}_task-mapper_run-{run}_space-{space}_hemi-{hemi}_desc-smoothed_bold.func.gii')
            else:
                d = op.join(dir_,
                            f'sub-{subject}_ses-{session}_task-mapper_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')
            d = surface.load_surf_data(d).T
            columns = pd.MultiIndex.from_product(
                [[hemi], np.arange(d.shape[1])], names=['hemi', 'vertex'])
            index = pd.MultiIndex.from_product(
                [[run], frametimes], names=['run', None])
            d_.append(pd.DataFrame(d, columns=columns, index=index))

        data.append(pd.concat(d_, 1))
    return pd.concat(data, 0)


def get_mapper_paradigm(subject, session, sourcedata, run=None):

    if run is None:
        run = 1

    events = pd.read_table(op.join(
        sourcedata, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_events.tsv'))

    events = events[events['trial_type'] == 'stimulation'].sort_values('onset')
    events['onset_halfway'] = events['onset']+events['duration'] / 2.
    events.index = pd.to_timedelta(events.onset_halfway, unit='s')
    tmp = pd.DataFrame([0, 0], columns=['n_dots'])
    n_vols = 125
    tr = get_tr(subject, session)
    tmp.index = pd.Index(pd.to_timedelta([0, (n_vols-1)*tr], 's'), name='time')
    paradigm = pd.concat((tmp, events)).n_dots.resample(
        '2.3S').nearest().to_frame('n_dots').astype(np.float32)
    paradigm['n_dots'] = np.log(paradigm['n_dots']).replace(-np.inf, -1e6)

    paradigm.index = pd.Index(get_frametimes(subject, session), name='time')

    return paradigm

def get_task_paradigm(subject=None, session=None, bids_folder='/data', run=None):

    if subject is None:
        return get_all_task_behavior(session=session, bids_folder=bids_folder)

    if run is None:
        runs = range(1,9)
    else:
        runs = [run]

    behavior = []

    for run in range(1, 9):
        b = pd.read_csv(op.join(bids_folder, f'sub-{subject}', f'ses-{session}',
                               'func', f'sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'), sep='\t')

        b['trial_nr'] = b['trial_nr'].astype(int)
        behavior.append(b.set_index('trial_nr'))

    behavior = pd.concat(behavior, keys=range(1,9), names=['run'])#.droplevel(1)
    # print(behavior)

    behavior = behavior.reset_index().set_index(
        ['run', 'trial_nr', 'trial_type'])
    stimulus1 = behavior.xs('stimulus 1', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'n1', 'prob1', 'n2', 'prob2']]
    stimulus1['duration'] = 0.6

    stimulus2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type']]
    stimulus2['duration'] = 0.6


    n1 = behavior.xs('stimulus 1', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'n1']]
    n1['duration'] = 0.6
    def zscore(n):
        return (n - n.mean()) / n.std()
    n1['modulation'] = zscore(n1['n1'])
    n1['trial_type'] = 'n_dots1'

    n2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'n2']]
    n2['duration'] = 0.6
    def zscore(n):
        return (n - n.mean()) / n.std()
    n2['modulation'] = zscore(n2['n2'])
    n2['trial_type'] = 'n_dots2'

    p1 = behavior.xs('stimulus 1', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'prob1']]
    p1 = p1[p1.prob1 == 1.0]
    p1['duration'] = 0.6
    p1['trial_type'] = 'certain1'

    p2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type', 'prob2']]
    p2 = p2[p2.prob2 == 1.0]
    p2['duration'] = 0.6
    p2['trial_type'] = 'certain2'

    events = pd.concat((stimulus1, stimulus2, n1, n2, p1, p2)).sort_index()
    events['modulation'].fillna(1.0, inplace=True)

    print(events)
    print(behavior.xs('certainty', 0, 'trial_type')['choice'].to_frame('certainty'))



    events['certainty'] = behavior.xs('certainty', 0, 'trial_type')['choice'].to_frame('certainty')

    return events


def get_task_behavior(subject, session, bids_folder='/data'):

    runs = range(1, 9)

    df = []

    for run in runs:
        d = pd.read_csv(op.join(bids_folder, f'sourcedata/sub-{subject}/behavior/ses-{session}/sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'), sep='\t')
        d = d[np.in1d(d.phase, [8,9])]
        d['trial_nr'] = d['trial_nr'].astype(int)
        d = d.pivot_table(index=['trial_nr'], values=['choice', 'certainty', 'n1', 'n2', 'prob1', 'prob2'])
        d['subject'], d['session'], d['scanner'], d['run'] = subject, session, session[:2], run
        d = d.set_index(['subject', 'session', 'run'], append=True).reorder_levels(['subject', 'session', 'run', 'trial_nr'])
        df.append(d)    

    df = pd.concat(df)

    df['task'] = 'task'
    df['log(n1)'] = np.log(df['n1'])

    df['log(risky/safe)'] = np.log(df['n1'] / df['n2'])

    ix = df.prob1 == 1.0
    df.loc[ix, 'log(risky/safe)'] = np.log(df.loc[ix, 'n2'] / df.loc[ix, 'n1'])
    df.loc[~ix, 'log(risky/safe)'] = np.log(df.loc[~ix, 'n1'] / df.loc[~ix, 'n2'])

    df.loc[ix, 'base_number'] = df.loc[ix, 'n1']
    df.loc[~ix, 'base_number'] = df.loc[~ix, 'n2']

    df['risky/safe'] = np.exp(df['log(risky/safe)'])

    df.loc[~ix, 'chose_risky'] = df.loc[~ix, 'choice'] == 1
    df.loc[ix, 'chose_risky'] = df.loc[ix, 'choice'] == 2
    df['chose_risky'] = df['chose_risky'].astype(bool)
    df['risky_first'] = df.prob1 == 0.55


    return df

def get_all_task_behavior(session=None, bids_folder='/data'):

    keys = []
    df = []

    subjects = get_all_subjects(bids_folder=bids_folder)

    if session is None:
        sessions = ['3t2', '7t2']
    else:
        sessions = [session]

    for subject, session in product(subjects, sessions):
        try:
            d = get_task_behavior(subject, session, bids_folder)
            df.append(d)
                
        except Exception as e:
            print(e)

    df = pd.concat(df)

    return df

def get_target_dir(subject, session, sourcedata, base, modality='func'):
    target_dir = op.join(sourcedata, 'derivatives', base, f'sub-{subject}', f'ses-{session}',
                         modality)

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    return target_dir


def write_gifti(subject, session, sourcedata, space, data, filename):
    dir_ = op.join(sourcedata, 'derivatives', 'fmriprep', f'sub-{subject}',
                   f'ses-{session}', 'func',)

    run = 1

    if session.endswith('1'):
        task = 'mapper'
    else:
        task = 'task'

    if data.ndim == 1:
        data = data.to_frame().T

    for hemi, d in data.groupby(['hemi'], axis=1):
        header = nb.load(op.join(dir_,
                                 f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_hemi-L_bold.func.gii')).header
        darrays = [nb.gifti.GiftiDataArray(
            data=d_.values) for _, d_ in d.iterrows()]
        im = gifti.GiftiImage(header=header,
                              darrays=darrays)
        im.to_filename(filename.format(hemi=hemi))


def get_volume_data(subject, session, run, sourcedata, smoothed=False, space='T1w'):

    if smoothed:
        raise NotImplementedError

    dir_ = op.join(sourcedata, 'derivatives', 'fmriprep', f'sub-{subject}',
                   f'ses-{session}', 'func',)

    if session[-1] == '1':
        task = 'mapper'
    elif session[-1] == '2':
        task = 'task'

    fn = op.join(
        dir_, f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_desc-preproc_bold.nii.gz')
    print(fn)

    return image.load_img(fn)


def get_brain_mask(subject, session, run, sourcedata, space='T1w', bold=True):

    if not bold:
        raise NotImplementedError

    dir_ = op.join(sourcedata, 'derivatives', 'fmriprep', f'sub-{subject}',
                   f'ses-{session}', 'func',)

    if session[-1] == '1':
        task = 'mapper'
    elif session[-1] == '2':
        task = 'task'

    fn = op.join(
        dir_, f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_desc-brain_mask.nii.gz')

    return image.load_img(fn)

def get_fs_subject(subject):

    if subject == 'fsaverage':
        return subject
    else:
        return f'sub-{subject}'


def get_surf_mask(subject, mask, hemi=None, bids_folder='/data'):

    if hemi is None:
        mask_l = get_surf_mask(subject, mask, 'L', bids_folder )
        mask_r = get_surf_mask(subject, mask, 'R', bids_folder )
        return pd.concat((mask_l, mask_r), axis=1, keys=['L', 'R'], names=['hemi'])
        

    fs_hemi = {'L':'lh', 'R':'rh'}[hemi]

    fs_subject = f'sub-{subject}'
    fn = op.join(bids_folder, 'derivatives', 'freesurfer', 
            get_fs_subject(subject), 'surf',
            f'{fs_hemi}.{mask}.mgz')

    d = surface.load_surf_data(fn).astype(np.bool)
    d = pd.Series(d, index=pd.Index(np.arange(len(d)), name='vertex'))
    return d




def get_prf_parameters_volume(subject, session, bids_folder,
        run=None,
        smoothed=False,
        cross_validated=True,
        hemi=None,
        mask=None,
        space='fsnative'):

    dir = 'encoding_model'
    if cross_validated:
        if run is None:
            raise Exception('Give run')

        dir += '.cv'

    if smoothed:
        dir += '.smoothed'

    parameters = []

    keys = ['mu', 'sd', 'amplitude', 'baseline']

    mask = get_volume_mask(subject, session, mask, bids_folder)
    masker = NiftiMasker(mask)

    for parameter_key in keys:
        if cross_validated:
            fn = op.join(bids_folder, 'derivatives', dir, f'sub-{subject}', f'ses-{session}', 
                    'func', f'sub-{subject}_ses-{session}_run-{run}_desc-{parameter_key}.optim_space-T1w_pars.nii.gz')
        else:
            fn = op.join(bids_folder, 'derivatives', dir, f'sub-{subject}', f'ses-{session}', 
                    'func', f'sub-{subject}_ses-{session}_desc-{parameter_key}.optim_space-T1w_pars.nii.gz')
        
        pars = pd.Series(masker.fit_transform(fn).ravel())
        parameters.append(pars)

    return pd.concat(parameters, axis=1, keys=keys, names=['parameter'])


def get_prf_parameters_surf(subject, session, bids_folder,
        run=None,
        smoothed=False,
        cross_validated=False,
        hemi=None,
        mask=None,
        space='fsnative'):

    if hemi is None:
        prf_l = get_prf_parameters(subject, session, bids_folder,
                run, smoothed, cross_validated, hemi='L',
                mask=mask, space=space)
        prf_r = get_prf_parameters(subject, session, bids_folder,
                run, smoothed, cross_validated, hemi='R',
                mask=mask, space=space)
        
        return pd.concat((prf_l, prf_r), axis=0, 
                keys=pd.Index(['L', 'R'], name='hemi'))


    dir = 'encoding_model'
    if cross_validated:
        if run is None:
            raise Exception('Give run')

        dir += '.cv'

    if smoothed:
        dir += '.smoothed'

    parameters = []

    keys = ['mu', 'sd', 'amplitude', 'baseline']

    if mask is not None:
        mask = get_surf_mask(subject, mask, hemi, bids_folder)

    for parameter_key in keys:
        if cross_validated:
            fn = op.join(bids_folder, 'derivatives', 'encoding_model.cv', f'sub-{subject}', f'ses-{session}', 
                    'func', f'sub-{subject}_ses-{session}_run-{run}_desc-{parameter_key}.optim_space-{space}_hemi-{hemi}.func.gii')
        else:
            fn = op.join(bids_folder, 'derivatives', 'encoding_model', f'sub-{subject}', f'ses-{session}', 
                    'func', f'sub-{subject}_ses-{session}_desc-{parameter_key}.optim_space-{space}_hemi-{hemi}.func.gii')

        pars = pd.Series(surface.load_surf_data(fn))
        pars.index.name = 'vertex'

        if mask is not None:
            pars = pars.loc[mask.values]

        parameters.append(pars)

    return pd.concat(parameters, axis=1, keys=keys, names=['parameter'])


def get_surf_distance_matrix(subject, mask, hemi=None, bids_folder='/data'):

    if subject == 'fsaverage':
        fs_subject = 'fsaverage'
    else:
        fs_subject = f'sub-{subject}'


    fs_hemi = {'L':'lh', 'R':'rh'}[hemi]

    if hemi is None:
        
        dist_matrix_l = get_surf_distance_matrix(subject, mask, 'L', bids_folder)
        dist_matrix_r = get_surf_distance_matrix(subject, mask, 'R', bids_folder)

        dist_matrix = pd.concat((dist_matrix_l, dist_matrix_r), axis=0).fillna(100)

    else:
        distance = op.join(bids_folder, 'derivatives', 'freesurfer', fs_subject, 'surf', f'{fs_hemi}.{mask}_distance.mgz')
        v = surface.load_surf_data(distance)

        mask = get_surf_mask(subject, mask, hemi, bids_folder)
        nlverts = int(mask.sum())
        
        dist_matrix = np.zeros((nlverts,nlverts))
        dist_matrix[np.triu_indices(nlverts, k = 1)] = v
        dist_matrix = dist_matrix + dist_matrix.T

        dist_matrix = pd.DataFrame(dist_matrix, index=mask.index[mask], columns=mask.index[mask])

        dist_matrix = pd.concat((dist_matrix,), axis=0, keys=[hemi], names=['hemi'])
        dist_matrix = pd.concat((dist_matrix,), axis=1, keys=[hemi], names=['hemi'])

        return dist_matrix


def get_volume_mask(subject, session, mask, bids_folder='/data'):

    if session.endswith('1'):
        task  = 'mapper'
    else:
        task  = 'task'

    base_mask = op.join(bids_folder, 'derivatives', f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-{task}_run-1_space-T1w_desc-brain_mask.nii.gz')

    if mask is None:
        return base_mask

    mask = mask.replace('_', '')
    mask = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-{mask}_mask.nii.gz')
    mask = image.resample_to_img(mask, base_mask, interpolation='nearest')

    return mask

def get_single_trial_volume(subject, session, mask=None, bids_folder='/data',
        smoothed=False):

    key= 'glm_stim1'

    if smoothed:
        key += '.smoothed'

    fn = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', f'ses-{session}', 'func', 
            f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims1_pe.nii.gz')

    im = image.load_img(fn)
    
    mask = get_volume_mask(subject, session, mask, bids_folder)
    paradigm = get_task_behavior(subject, session, bids_folder)
    masker = NiftiMasker(mask_img=mask)

    data = pd.DataFrame(masker.fit_transform(im), index=paradigm.index)

    return data


