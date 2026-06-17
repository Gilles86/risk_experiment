import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scipy.stats as ss


def get_subjectwise_priors(behavior, values=None):

    if values is None:
        values = np.arange(5, 28*4+1)

    priors_subjectwise = behavior.groupby(['subject', 'session'])['n1'].apply(lambda x: pd.Series(ss.gaussian_kde(x, bw_method=None)(values),
                                                                                              index=values)).unstack()
    priors_subjectwise = priors_subjectwise.div(priors_subjectwise.sum(1), axis=0)
    priors_subjectwise.columns.name = 'n'

    return priors_subjectwise

def get_summary_pars(pdf, behavior, method='objective_smooth_natural', values=None):

    if method != 'objective_smooth_natural':
        raise NotImplementedError

    if values is None:
        values = np.arange(5, 28*4+1)
        
    priors = get_subjectwise_priors(behavior, values)

    pdf.columns = pdf.columns.astype(np.float32)

    f = interp1d(pdf.columns, pdf)
    pdf = pd.DataFrame(f(np.log(values)), index=pdf.index, columns=pd.Index(values, name='n'))
    
    pdf = pdf * priors
    
    pdf = pdf.div(pdf.sum(1), axis=0)

    E = pd.Series((pdf*pdf.columns.astype(float)).sum(1)).to_frame('E').join(behavior[['n1', 'prob1', 'n_safe', 'risky_first']].droplevel("run")).reorder_levels(pdf.index.names).sort_index()

    E['map'] = pdf.idxmax(1).astype(float)
    E['sd'] = pd.Series((pdf.values * np.abs(E['E'].values[:, np.newaxis] - pdf.columns.values.astype(float)[np.newaxis, :])).sum(1), index=pdf.index)
    E['error'] = E['E'] - E['n1']
    E['abs(error)'] = E['error'].abs()    

    return E