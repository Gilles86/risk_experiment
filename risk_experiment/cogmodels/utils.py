import pandas as pd
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from risk_experiment.utils import get_all_subjects


def cluster_offers(d, n=6, key='log(risky/safe)'):
    return pd.qcut(d[key], n, duplicates='drop').apply(lambda x: x.mid)



def plot_ppc(df, ppc, plot_type=1, var_name='ll_bernoulli', level='subject', col_wrap=5):

    assert (var_name in ['p', 'll_bernoulli'])

    ppc = ppc.xs(var_name, 0, 'variable').copy()

    df = df.copy()

    if level == 'group':
        df['log(risky/safe)'] = df['bin(risky/safe)']
        ppc = ppc.reset_index('log(risky/safe)')
        ppc['log(risky/safe)'] = ppc.index.get_level_values('bin(risky/safe)')

    if plot_type == 1:
        groupby = ['risky_first', 'log(risky/safe)']
    elif plot_type in [2, 4]:
        groupby = ['risky_first', 'n_safe']
    elif plot_type in [3, 5]:
        groupby = ['risky_first', 'n_safe', 'log(risky/safe)']
    elif plot_type in [6]:
        groupby = ['uncertainty', 'log(risky/safe)']
    elif plot_type in [7]:
        groupby = ['risky_first', 'uncertainty', 'log(risky/safe)']
    elif plot_type in [8]:
        groupby = ['risky_first', 'uncertainty', 'n_safe']
    elif plot_type in [9]:
        groupby = ['risky_first', 'median_split_uncertainty', 'log(risky/safe)']
    elif plot_type in [10]:
        groupby = ['risky_first', 'median_split_uncertainty', 'n_safe']
    elif plot_type in [11]:
        groupby = ['median_split(sd)', 'log(risky/safe)']
    elif plot_type in [12]:
        groupby = ['risky_first', 'median_split(sd)', 'n_safe']
    elif plot_type in [13]:
        groupby = ['median_split_pupil_baseline', 'log(risky/safe)']
    elif plot_type in [14]:
        groupby = ['risky_first', 'median_split_pupil_baseline', 'n_safe']
    elif plot_type in [15]:
        groupby = ['risky_first', 'median_split_pupil_baseline', 'log(risky/safe)']
    elif plot_type in [16]:
        groupby = ['risky_first', 'median_split(sd)', 'log(risky/safe)']
    else:
        raise NotImplementedError

    if level == 'group':
        ppc = ppc.groupby(['subject']+groupby).mean()

    if level == 'subject':
        groupby = ['subject'] + groupby

    # print(ppc)
    ppc_summary = summarize_ppc(ppc, groupby=groupby)
    print(ppc_summary)
    p = df.groupby(groupby).mean()[['chose_risky']]
    print(p)
    # ppc_summary = pd.concat((p, ppc_summary), axis=1).sort_index()
    ppc_summary = ppc_summary.join(p).reset_index()

    print(ppc_summary)

    if 'risky_first' in groupby:
        ppc_summary['Order'] = ppc_summary['risky_first'].map({True:'Risky first', False:'Safe first'})

    if 'n_safe' in groupby:
        ppc_summary['Safe offer'] = ppc_summary['n_safe'].astype(int)

    ppc_summary['Prop. chosen risky'] = ppc_summary['chose_risky']

    if 'log(risky/safe)' in groupby:
        if level == 'group':
            ppc_summary['Predicted acceptance'] = ppc_summary['log(risky/safe)']
        else:
            ppc_summary['Log-ratio offer'] = ppc_summary['log(risky/safe)']

    if plot_type in [2, 7]:
            x = 'Safe offer'
    else:
        if level == 'group':
            x = 'Predicted acceptance'
        else:
            x = 'Log-ratio offer'

    if plot_type in [1, 2]:
        fac = sns.FacetGrid(ppc_summary,
                            col='subject' if level == 'subject' else None,
                            hue='Order',
                            col_wrap=col_wrap if level == 'subject' else None)

    elif plot_type == 3:
        fac = sns.FacetGrid(ppc_summary,
                            col='Safe offer',
                            hue='Order',
                            row='subject' if level == 'subject' else None)
    elif plot_type == 4:


        if level == 'group':
            rnp = df.groupby(['subject'] + groupby, group_keys=False).apply(get_rnp).to_frame('rnp')
            rnp = rnp.groupby(groupby).mean()
        else:
            rnp = df.groupby(groupby, group_keys=False).apply(get_rnp).to_frame('rnp')

        print(ppc_summary)
        print(rnp)
        ppc_summary = ppc_summary.join(rnp)
        print(ppc_summary)
        fac = sns.FacetGrid(ppc_summary,
                            hue='Order',
                            col='subject' if level == 'subject' else None,
                            col_wrap=col_wrap if level == 'subject' else None)

        print(ppc_summary)
        fac.map_dataframe(plot_prediction, x='Safe offer', y='p_predicted')
        fac.map(plt.scatter, 'Safe offer', 'rnp')
        fac.map(lambda *args, **kwargs: plt.axhline(.55, c='k', ls='--'))

    elif plot_type == 5:
        fac = sns.FacetGrid(ppc_summary,
                            col='Order',
                            hue='Safe offer',
                            row='subject' if level == 'subject' else None,
                            palette='coolwarm')
    elif plot_type == 6:
        fac = sns.FacetGrid(ppc_summary,
                            hue='uncertainty',
                            row='subject' if level == 'subject' else None,
                            aspect=2.,
                            palette=sns.color_palette('viridis', n_colors=4))
    elif plot_type in [7, 8]:
        fac = sns.FacetGrid(ppc_summary,
                            col='Order',
                            hue='uncertainty',
                            row='subject' if level == 'subject' else None,
                            palette=sns.color_palette('viridis', n_colors=4))
    elif plot_type in [9, 10]:
        fac = sns.FacetGrid(ppc_summary,
                            col='Order',
                            hue='median_split_uncertainty',
                            row='subject' if level == 'subject' else None,
                            palette=sns.color_palette('viridis', n_colors=2))

    elif plot_type == 11:
        fac = sns.FacetGrid(ppc_summary,
                            hue='median_split(sd)',
                            col='subject' if level == 'subject' else None,
                            palette=sns.color_palette()[2:],
                            col_wrap=col_wrap if level == 'subject' else None)
    elif plot_type == 12:
        x = 'n_safe'
        fac = sns.FacetGrid(ppc_summary,
                            hue='median_split(sd)',
                            col='Order',
                            palette=sns.color_palette()[2:],
                            row='subject' if level == 'subject' else None,)
    elif plot_type == 13:
        fac = sns.FacetGrid(ppc_summary,
                            hue='median_split_pupil_baseline',
                            col='subject' if level == 'subject' else None,
                            palette=sns.color_palette()[4:],
                            col_wrap=col_wrap if level == 'subject' else None)
    elif plot_type == 14:
        x = 'n_safe'
        fac = sns.FacetGrid(ppc_summary,
                            hue='median_split_pupil_baseline',
                            col='Order',
                            palette=sns.color_palette()[4:],
                            row='subject' if level == 'subject' else None,)
    elif plot_type == 15:
        fac = sns.FacetGrid(ppc_summary,
                            hue='median_split_pupil_baseline',
                            col='Order',
                            palette=sns.color_palette()[4:],
                            row='subject' if level == 'subject' else None,)
    elif plot_type == 16:
        fac = sns.FacetGrid(ppc_summary,
                            hue='median_split(sd)',
                            col='Order',
                            palette=sns.color_palette()[2:],
                            row='subject' if level == 'subject' else None,)

    print("X", x)
    if plot_type in [1,2,3, 5, 11, 12, 13, 14, 15, 16]:
        fac.map_dataframe(plot_prediction, x=x)
        fac.map(plt.scatter, x, 'Prop. chosen risky')
        fac.map(lambda *args, **kwargs: plt.axhline(.5, c='k', ls='--'))

    if plot_type in [1, 3, 5, 11, 13, 15, 16]:
        if level == 'subject':
            fac.map(lambda *args, **kwargs: plt.axvline(np.log(1./.55), c='k', ls='--'))
        else:
            fac.map(lambda *args, **kwargs: plt.axvline(3.5, c='k', ls='--'))
            plt.xticks([])

    
    fac.add_legend()

    return fac

def plot_prediction(data, x, color, y='p_predicted', alpha=.25, **kwargs):
    data = data[~data['hdi025'].isnull()]

    plt.fill_between(data[x], data['hdi025'],
                     data['hdi975'], color=color, alpha=alpha)
    plt.plot(data[x], data[y], color=color)


def summarize_ppc(ppc, groupby=None):

    if groupby is not None:
        ppc = ppc.groupby(groupby).mean()

    e = ppc.mean(1).to_frame('p_predicted')
    hdi = pd.DataFrame(az.hdi(ppc.T.values), index=ppc.index,
                       columns=['hdi025', 'hdi975'])

    print(hdi)
    return pd.concat((e, hdi), axis=1)

def get_rnp(d):
    return (1./d['frac']).quantile(d.chose_risky.mean())


def format_bambi_ppc(trace, model, df):

    preds = []
    for key, kind in zip(['ll_bernoulli', 'p'], ['pps', 'mean']):
        pred = model.predict(trace, kind=kind, inplace=False) 
        if kind == 'pps':
            pred = pred['posterior_predictive']['chose_risky'].to_dataframe().unstack(['chain', 'draw'])['chose_risky']
        else:
            pred = pred['posterior']['chose_risky_mean'].to_dataframe().unstack(['chain', 'draw'])['chose_risky_mean']
        pred.index = df.index
        pred = pred.set_index(pd.MultiIndex.from_frame(df), append=True)
        preds.append(pred)

    pred = pd.concat(preds, keys=['ll_bernoulli', 'p'], names=['variable'])
    return pred


def plot_subjectwise_posterior(trace, key, hue=None, ref_value=None, color=None, palette=sns.color_palette()):
    d = trace.posterior[key].to_dataframe()

    if ref_value is not None:
        if isinstance(ref_value, pd.DataFrame):
            print('yooo')
            d = d.join(ref_value)

    print(d)

    if (color is None) and (hue is None):
        color = palette[0]

    fac = sns.catplot(x='subject', y=key, hue=hue, data=d.reset_index(),
    kind='violin', aspect=8, color=color, palette=palette if hue is not None else None)

    if ref_value is not None:
        print('yoooo')
        fac.map(sns.pointplot, 'subject', 'ref_value', color='k')

    # if ref_value:
    #     fac.map(lambda *args, **kwargs: plt.axhline(ref_value, c='k', ls='--'))

    return fac

def plot_groupwise_posterior(trace, key, hue, ref_value, palette=sns.color_palette(), color=None):

    d = trace.posterior[key+'_mu'].to_dataframe()

    if ref_value is not None:
        if isinstance(ref_value, pd.DataFrame):
            print('yooo')
            d = d.join(ref_value)
        else:
            d['ref_value'] = ref_value

    if (color is None) and (hue is None):
        color = palette[0]

    # if hue is not None:
    #     fac = sns.catplot(x=hue, y=key+'_mu', data=d.reset_index(), kind='violin', aspect=1., palette=palette)
    # else:
    fac = sns.FacetGrid(d.reset_index(), hue=hue)
    fac.map(sns.kdeplot, key+'_mu', fill=True)
    # fac.map(sns.histplot, key+'_mu', stat='density')

    if ref_value is not None:
        print(f'REF VALUE {ref_value}')
        fac.map(lambda *args, **kwargs: plt.axvline(ref_value.values, c='k', ls='--'))

    if hue is not None:
        fac.add_legend()

    fac.set(ylabel=f'p({key})')

    return fac

def invprobit(x):
    return ss.norm.ppf(x)

def extract_intercept_gamma(trace, model, data, group=False):

    fake_data = get_fake_data(data, group)

    pred = model.predict(trace, 'mean', fake_data, inplace=False, include_group_specific=not group)['posterior']['chose_risky_mean']

    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake_data))

    # return pred

    pred0 = pred.xs(0, 0, 'x')
    intercept = pd.DataFrame(invprobit(pred0), index=pred0.index, columns=pred0.columns)
    gamma = invprobit(pred.xs(1, 0, 'x')) - intercept

    intercept = pd.concat((intercept.droplevel(0, 1),), keys=['intercept'], axis=1)
    gamma = pd.concat((gamma.droplevel(0, 1),), keys=['gamma'], axis=1)

    return intercept, gamma

def get_fake_data(data, group=False):

    data = data.reset_index()

    if group:
        permutations = [[1]]
    else:
        permutations = [data['subject'].unique()]


    permutations += [np.array([0., 1.]), data['n_safe'].unique(), [False, True]]
    names=['subject', 'x', 'n_safe', 'risky_first']

    if 'sd' in data.columns:
        permutations += [[data['sd'].mean()]]
        names += ['sd']

    if 'median_split_sd' in data.columns:
        permutations += [[True, False]]
        names += ['median_split_sd']

    if 'median_split_pupil_baseline' in data.columns:
        permutations += [data['median_split_pupil_baseline'].unique()]
        names += ['median_split_pupil_baseline']

    if 'pupil' in data.columns:
        permutations += [[data['pupil'].mean()]]
        names += ['pupil']

    print(names)
    fake_data = pd.MultiIndex.from_product(permutations, names=names).to_frame().reset_index(drop=True)

    return fake_data
