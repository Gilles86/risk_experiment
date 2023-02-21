import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as op
import argparse
from fit_probit import build_model, get_data
import arviz as az
from utils import extract_intercept_gamma, plot_ppc, format_bambi_ppc, cluster_offers
import pandas as pd


def main(model_label, session, bids_folder='/data/ds-risk', col_wrap=5, only_ppc=False):


    df = get_data(model_label, session)
    model = build_model(model_label, df)

    idata = az.from_netcdf(op.join(bids_folder, f'derivatives/cogmodels/model-{model_label}_ses-{session}_trace.netcdf'))

    target_folder = op.join(bids_folder, f'derivatives/cogmodels/figures/{model_label}/ses-{session}')
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    intercept_group, gamma_group = extract_intercept_gamma(idata, model, df, True)
    rnp_group = get_rnp(intercept_group, gamma_group)

    gamma_group['Order'] = gamma_group.index.get_level_values('risky_first').map({True:'Risky first', False:'Safe first'})
    rnp_group['Order'] = gamma_group['Order']
    rnp_group.set_index('Order', inplace=True, append=True)
    gamma_group.set_index('Order', inplace=True, append=True)

    intercept, gamma = extract_intercept_gamma(idata, model, df, False)
    rnp = np.clip(get_rnp(intercept, gamma), 0, 1)

    gamma['Order'] = gamma.index.get_level_values('risky_first').map({True:'Risky first', False:'Safe first'})
    rnp['Order'] = gamma['Order']
    rnp.set_index('Order', inplace=True, append=True)
    gamma.set_index('Order', inplace=True, append=True)

    print(f'*** only_ppc: {only_ppc}')
    if not only_ppc:
        if model_label.startswith('probit_simple'):
            # Violin plot of conditions
            # gamma_groupfig = sns.catplot(gamma_group.stack().stack().reset_index(), y='gamma', x='Order', hue='stimulation_condition', kind='violin')
            gamma_groupfig = sns.kdeplot(gamma_group.stack().stack())
            plt.savefig(op.join(target_folder, 'group_pars_gamma.pdf'))

            plt.figure()
            gamma_groupfig = sns.kdeplot(rnp_group.stack().stack())
            plt.savefig(op.join(target_folder, 'rnp_pars_gamma.pdf'))

            sns.catplot(gamma.stack([-2, -1]).reset_index(), x='subject', y='gamma', kind='violin', aspect=2.)
            plt.savefig(op.join(target_folder, 'subject_par-gamma.pdf'))

            sns.catplot(rnp.stack([-2, -1]).reset_index(), x='subject', y='rnp', kind='violin', aspect=2.)
            plt.savefig(op.join(target_folder, 'subject_par-rnp.pdf'))

        elif model_label.startswith('probit_order'):
            # Violin plot of conditions
            gamma_groupfig = sns.catplot(gamma_group.stack().stack().reset_index(), y='gamma', x='Order', hue='stimulation_condition', kind='violin')
            plt.savefig(op.join(target_folder, 'group_pars_gamma.pdf'))

            rnp_groupfig = sns.catplot(rnp_group.stack().stack().reset_index(), y='rnp',
            x='Order', hue='stimulation_condition', kind='violin')
            plt.savefig(op.join(target_folder, 'group_pars_rnp.pdf'))
            plt.axhline(0.55, c='k', ls='--')

        elif model_label.startswith('probit_full'):
            # Violin plot of conditions
            gamma_groupfig = sns.catplot(gamma_group.stack().stack().reset_index(), y='gamma', x='n_safe',
            col='Order',
            hue='stimulation_condition', kind='violin')

            # Distribution of difference
            plt.figure()
            gamma_diff = gamma_group.stack([0, 1, 2]).unstack(['stimulation_condition'])
            gamma_diff = (gamma_diff['ips'] - gamma_diff['vertex']).to_frame('gamma_diff')
            print(gamma_diff)

            fac = sns.FacetGrid(gamma_diff.reset_index(), col='n_safe', hue='Order', palette=sns.color_palette()[2:])
            fac.map(sns.distplot, 'gamma_diff')
            fac.map(lambda *args, **kwargs: plt.axvline(0.0, c='k', ls='--'))
            fac.add_legend()
            fac.savefig(op.join(target_folder, 'group_gamma_diff.pdf'))
            plt.close()

            plt.figure()
            rnp_diff = rnp_group.stack([0, 1, 2]).unstack(['stimulation_condition'])
            rnp_diff = (rnp_diff['ips'] - rnp_diff['vertex']).to_frame('rnp_diff')

            fac = sns.FacetGrid(rnp_diff.reset_index(), hue='Order', col='n_safe', palette=sns.color_palette()[2:])
            fac.map(sns.distplot, 'rnp_diff')
            fac.map(lambda *args, **kwargs: plt.axvline(0.0, c='k', ls='--'))
            fac.add_legend()
            plt.savefig(op.join(target_folder, 'group_rnp_diff.pdf'))
            plt.close()

            rnp_groupfig = sns.catplot(rnp_group.stack().stack().reset_index(), y='rnp',
            x='n_safe', col='Order', hue='stimulation_condition', kind='violin')
            rnp_groupfig.map(lambda *args, **kwargs: plt.axhline(0.55, c='k', ls='--'))


            # Subjectwise parameter plots
            gamma_fig = sns.catplot(gamma.stack([1, 2]).reset_index(), x='subject', y='gamma', row='Order', hue='n_safe', col='stimulation_condition', kind='violin', aspect=3.)
            rnp_fig = sns.catplot(rnp.stack([1, 2]).reset_index(), x='subject', y='rnp', row='Order', hue='n_safe', col='stimulation_condition', kind='violin', aspect=3.)
            rnp_fig.map(lambda *arsg, **kwargs: plt.axhline(0.55, c='k', ls='--'))

            plt.axhline(0.55, c='k', ls='--')
        elif model_label.startswith('probit_neural') or model_label.startswith('probit_pupil'):
            if model_label in ['probit_neural1', 'probit_neural2', 'probit_neural4']:
                keys = ['sd', 'x:sd']
            elif model_label in ['probit_neural3', 'probit_neural5', 'probit_neural6', 'probit_neural7']:
                keys = ['median_split_sd', 'x:median_split_sd']
            elif model_label in ['probit_pupil1']:
                keys = ['median_split_pupil_baseline', 'x:median_split_pupil_baseline']

            for key in keys:
                trace = idata.posterior[key].to_dataframe()
                plt.figure()
                sns.kdeplot(trace, fill=True)
                plt.title(f'p < 0.0: {(trace < 0.0).mean().values[0]:0.03f}')
                plt.savefig(op.join(target_folder, f'group_{key}_sd.pdf'))
                plt.savefig(op.join(target_folder, f'group_{key}_sd.png'))

        else:
            raise NotImplementedError


    # PPC

    df['log(risky/safe)'] = df.groupby(['subject'],
                                            group_keys=False).apply(cluster_offers)
    ppc = format_bambi_ppc(idata, model, df)

    plot_types = [1,2,3, 5]

    if model_label.startswith('probit_neural'):
        plot_types += [11, 12]

    if model_label.startswith('probit_neural'):
        plot_types += [11, 12]

    plot_types = [11, 12]

    for plot_type in plot_types:
        for var_name in ['p', 'll_bernoulli']:
            for level in ['group', 'subject']:
                target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'figures', model_label, f'ses-{session}', level, var_name)

                if not op.exists(target_folder):
                    os.makedirs(target_folder)

                fig = plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name, col_wrap=col_wrap)

                fn = f'{level}_ses-{session}_plot-{plot_type}_model-{model_label}_pred.pdf'
                fig.savefig(op.join(target_folder, fn))

                fn = f'{level}_ses-{session}_plot-{plot_type}_model-{model_label}_pred.png'
                fig.savefig(op.join(target_folder, fn))

def get_rnp(intercept, gamma):
    rnp = np.exp(intercept['intercept']/gamma['gamma'])

    rnp = pd.concat((rnp, ), keys=['rnp'], axis=1)
    return rnp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('session')
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    parser.add_argument('--only_ppc', action='store_true')
    args = parser.parse_args()

    main(args.model_label, session=args.session, bids_folder=args.bids_folder, only_ppc=args.only_ppc)