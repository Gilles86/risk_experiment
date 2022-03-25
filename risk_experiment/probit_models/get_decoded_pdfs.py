from nilearn import surface
import pandas as pd
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pingouin
import seaborn as sns
from risk_experiment.utils import get_task_paradigm
from tqdm.contrib.itertools import product

bids_folder = '/data'


keys = []
pdfs = []

for subject, session, smoothed, n_voxels, mask in product(['{:02d}'.format(i) for i in range(2, 33)], ['3t2', '7t2'], [False],  [250],
                                                         ['wang15_ips', 'wang15_ipsL', 'wang15_ipsR', 'npcl', 'npcr', 'npc']):
    
    key= 'decoded_pdfs.volume'
    if smoothed:
        key += '.smoothed'        
    
    try:
        pdf = pd.read_csv(op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func', f'sub-{subject}_ses-{session}_mask-{mask}_nvoxels-{n_voxels}_space-T1w_pars.tsv'), sep='\t',index_col=0)

        paradigm = get_task_paradigm(subject, session, bids_folder).query('trial_type == "stimulus 1"')
        paradigm['log(n1)'] = np.log(paradigm['n1'])
        paradigm['trial_nr'] = np.arange(1, 193)
        paradigm.set_index('trial_nr', append=True, inplace=True)

        pdf.index = paradigm.index
        pdf = pd.concat((paradigm, pdf), axis=1, keys=['paradigm', 'pdf'])
        
        
        keys.append((subject, session, {False:'unsmoothed', True:'smoothed'}[smoothed], n_voxels, mask))
        pdfs.append(pdf)
    except Exception as e:
        print(e)
#         pass


pdf = pd.concat(pdfs, keys=keys, names=['subject', 'session', 'smoothed', 'n_voxels', 'mask'])

E = pd.Series((pdf['pdf'].values * pdf['pdf'].columns.values[np.newaxis, :].astype(float)).sum(1),
              index=pdf.index).to_frame('E')


E['log(n1)'] = pdf[('paradigm', 'log(n1)')]
E['n1'] = pdf[('paradigm', 'n1')]
E['prob1'] = pdf[('paradigm', 'prob1')]
E['error'] = E['E'] - E['log(n1)']
E['abs(error)'] = E['error'].abs()
E['sd'] = pd.Series(np.sqrt((pdf['pdf'].values * (E['E'].values[:, np.newaxis] - pdf['pdf'].columns.values.astype(float)[np.newaxis, :])**2).sum(1)),
               index=pdf.index)

E['map'] = pdf['pdf'].idxmax(1).astype(np.float)

for mask, e in E.groupby('mask'):
    e.to_csv(op.join(bids_folder, f'derivatives/decoded_pdfs.volume/group_mask-{mask}_pars.tsv'), sep='\t')
