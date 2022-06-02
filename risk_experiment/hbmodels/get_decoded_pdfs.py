from nilearn import surface
import pandas as pd
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pingouin
import seaborn as sns
from risk_experiment.utils import get_task_paradigm, get_all_task_behavior
from tqdm.contrib.itertools import product
from scipy.interpolate import interp1d
import scipy.stats as ss

bids_folder = '/data'


keys = []
pdfs = []

masks = ['npcr']

behavior = get_all_task_behavior(bids_folder=bids_folder)

for subject, session, smoothed, n_voxels, mask in product(['{:02d}'.format(i) for i in range(2, 33)], ['3t2', '7t2'], [False],  [250],
        ['wang15_ips', 'wang15_ipsL', 'wang15_ipsR', 'npcl', 'npcr', 'npc']):
    
    key= 'decoded_pdfs.volume'
    if smoothed:
        key += '.smoothed'        
    
    try:
        pdf = pd.read_csv(op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func', f'sub-{subject}_ses-{session}_mask-{mask}_nvoxels-{n_voxels}_space-T1w_pars.tsv'), sep='\t',index_col=0)


        pdf.index = behavior.loc[(subject, session)].index
        keys.append((subject, session, {False:'unsmoothed', True:'smoothed'}[smoothed], n_voxels, mask))
        pdfs.append(pdf)
    except Exception as e:
        print(e)
        # pass


pdf = pd.concat(pdfs, keys=keys, names=['subject', 'session', 'smoothed', 'n_voxels', 'mask'])

pdf.columns = pdf.columns.astype(np.float32)
f = interp1d(pdf.columns, pdf)
x = np.log(np.arange(5, 28*4+1))
pdf = pd.DataFrame(f(x),
                    index=pdf.index,
                    columns=pd.Index(x, name='n'))

print(pdf)

pdf.columns = np.round(np.exp(pdf.columns.astype(float))).astype(int)

values = pdf.columns.astype(np.float32)
priors_subjectwise = behavior.groupby(['subject', 'session'])['n1'].apply(lambda x: pd.Series(ss.gaussian_kde(x, bw_method=None)(values),
                                                                                              index=values)).unstack()
priors_subjectwise = priors_subjectwise.div(priors_subjectwise.sum(1), axis=0)



pdf = pdf*priors_subjectwise
pdf = pdf.div(pdf.sum(1), axis=0)

E = pd.Series((pdf*pdf.columns.astype(float)).sum(1)).to_frame('E')
E = E.join(behavior[['n1', 'log(n1)', 'prob1', 'risky_first']].droplevel("run")).reorder_levels(pdf.index.names).sort_index()
E['map'] = pdf.idxmax(1).astype(float)
E['error'] = E['E'] - E['n1']
E['abs(error)'] = E['error'].abs()
E['sd1'] = (pdf*np.abs((E['E'].values[:, np.newaxis] - pdf.columns.values.astype(float)[np.newaxis, :]))).sum(1)
E['sd2'] = np.sqrt((pdf*(E['E'].values[:, np.newaxis] - pdf.columns.values.astype(float)[np.newaxis, :])**2).sum(1))

for mask, e in E.groupby('mask'):
    e.to_csv(op.join(bids_folder, f'derivatives/decoded_pdfs.volume/group_mask-{mask}_pars.tsv'), sep='\t')
