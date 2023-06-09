import pandas as pd
import pingouin
import matplotlib.pyplot as plt
import seaborn
import os
import os.path as op
import argparse
import arviz as az
from fit_model import get_data, build_model
import numpy as np
import seaborn as sns
from bauer.utils.bayes import softplus, logistic
from utils import plot_ppc, cluster_offers
from bauer.models import RiskModel, RiskRegressionModel, RiskLapseModel, RiskLapseRegressionModel, RegressionModel, ExpectedUtilityRiskModel, ExpectedUtilityRiskRegressionModel



def main(model_label, session, bids_folder='/data/ds-risk', col_wrap=5, plot_traces=False, group_only=False, only_ppc=False, roi=None):

    df = get_data(model_label, session, bids_folder, roi=roi)
    model = build_model(model_label, df, roi=roi)
    model.build_estimation_model()

    print(issubclass(type(model), RiskRegressionModel))

    roi_str = f'_{roi}' if roi is not None else ''    
    session_str = f'/ses-{session}' if session is not None else ''

    if session is None:
        idata = az.from_netcdf(op.join(bids_folder, 'derivatives', 'cogmodels', f'model-{model_label}{roi_str}_trace.netcdf'))
    else:
        idata = az.from_netcdf(op.join(bids_folder, 'derivatives', 'cogmodels', f'ses-{session}_model-{model_label}{roi_str}_trace.netcdf'))

    target_folder = op.join(bids_folder, f'derivatives/cogmodels/figures/{model_label}{roi_str}{session_str}')
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    if plot_traces and (not only_ppc):
        plot_trace(idata, target_folder)

    if not only_ppc: 
        plot_parameters(model, idata,  target_folder, session, df, model_label)

    if not (df.groupby(['subject', 'log(risky/safe)']).size().groupby('subject').size() < 7).all():
        df['log(risky/safe)'] = df.groupby(['subject'],
                                        group_keys=False).apply(cluster_offers)

    if 'session' in df.columns:
        df = df.droplevel('session', axis=0)

    print(df)
    ppc = model.ppc(trace=idata.sel(draw=slice(None, None, 10)), data=df)

    # "Chose risky" vs "chose 2nd option coding"
    ppc.loc[ppc.index.get_level_values('risky_first')] = 1 - ppc.loc[ppc.index.get_level_values('risky_first')]

    plot_ppcs(model_label, ppc, df, bids_folder, session, col_wrap, group_only)

def plot_trace(idata, target_folder):
    az.plot_trace(idata, var_names=['~p'])
    plt.savefig(op.join(target_folder, 'traces.pdf'))


def plot_parameters(model, idata, target_folder, session, df, model_label):
    def plot_parameter(par, regressor, trace, transform=False, **kwargs):
        t = trace.copy()
        print(regressor, t)

        if (par in ['prior_std', 'risky_prior_std', 'safe_prior_std', 'n1_evidence_sd', 'n2_evidence_sd', 'evidence_sd']) & (regressor == 'Intercept') & transform:
            t = softplus(t)

        if (par in ['p_lapse']) & (regressor == 'Intercept') & transform:
            t = logistic(t)

        sns.kdeplot(t, fill=True, **kwargs)
        if regressor != 'Intercept':
            plt.axvline(0.0, c='k', ls='--')
            txt = f'p({par} < 0.0) = {np.round((t.values < 0.0).mean(), 3)}'
            plt.xlabel(txt)

        else:
            if par == 'risky_prior_mu':
                plt.axvline(np.log(df['n_risky']).mean(), c='k', ls='--')
            elif par == 'risky_prior_std':
                plt.axvline(np.log(df['n_risky']).std(), c='k', ls='--')
            elif par == 'safe_prior_mu':
                for n_safe in np.log([7., 10., 14., 20., 28.]):
                    plt.axvline(n_safe, c='k', ls='--')

                plt.axvline(np.log(df['n_safe']).mean(), c='k', ls='--', lw=2)
            elif par == 'safe_prior_std':
                plt.axvline(np.log(df['n_safe']).std(), c='k', ls='--')


    for par in model.free_parameters:

        trace = idata.posterior[par+'_mu'].to_dataframe()

        if type(model) in [RiskModel, RiskLapseModel, ExpectedUtilityRiskModel]:
            plt.figure()
            plot_parameter(par, 'Intercept', trace, transform=False)
            plt.savefig(op.join(target_folder, f'group_par-{par}.Intercept.pdf'))
            plt.savefig(op.join(target_folder, f'group_par-{par}.Intercept.png'))
            plt.close()
        elif type(model) in [RiskRegressionModel, RiskLapseRegressionModel, ExpectedUtilityRiskRegressionModel]:
            for regressor, t in trace.groupby(par+'_regressors'):
                plt.figure()
                plot_parameter(par, regressor, t, transform=True)

                plt.savefig(op.join(target_folder, f'group_par-{par}.{regressor}.png'))
                plt.savefig(op.join(target_folder, f'group_par-{par}.{regressor}.pdf'))
                plt.close()

        plt.close()

    if (model_label == '2') or (model_label.startswith('4')):
        pairs, names, palettes = [('n1_evidence_sd', 'n2_evidence_sd')], ['evidence_sd'], [sns.color_palette()]
    elif model_label.startswith('rnp'):
        pairs, names, palettes = [], [], []
    else:
        pairs, names, palettes = [('n1_evidence_sd', 'n2_evidence_sd'), ('risky_prior_mu', 'safe_prior_mu'), ('risky_prior_std', 'safe_prior_std')], ['evidence_sd', 'prior_mu', 'prior_std'], [sns.color_palette(), sns.color_palette("RdBu", 2), sns.color_palette("RdBu", 2)]


    # SUBJECTWISE
    for pair, name, palette in zip(pairs, names, palettes):
        plt.figure(figsize=(20, 4))
        if type(model) is RiskModel:
            d = pd.concat((idata.posterior[pair[0]].to_dataframe(), idata.posterior[pair[1]].to_dataframe()), keys=pair, names=['Variable']).stack().to_frame('Value')
            g = sns.violinplot(d.reset_index(), x='subject', y='Value', hue='Variable', aspect=4, palette=palette)
            # g.add_legend()
            plt.suptitle(session)
            plt.savefig(op.join(target_folder, f'subject_par-{name}.Intercept.pdf'))
            plt.savefig(op.join(target_folder, f'subject_par-{name}.Intercept.png'))
            plt.close()
        elif type(model) is RiskRegressionModel:
            pass


    # GROUPWISE
    for pair, name, palette in zip(pairs, names, palettes):
        plt.figure()
        if type(model) is RiskModel:
            d = pd.concat((idata.posterior[pair[0]+'_mu'].to_dataframe(), idata.posterior[pair[1]+'_mu'].to_dataframe()), keys=pair, names=['Variable']).stack().to_frame('Value')

            g = sns.FacetGrid(d.reset_index(), hue='Variable', palette=palette)
            g.map(sns.kdeplot, 'Value', fill=True)
            g.add_legend()

            g.fig.suptitle(session)
            plt.savefig(op.join(target_folder, f'group_par-{name}.Intercept.pdf'))
            plt.savefig(op.join(target_folder, f'group_par-{name}.Intercept.png'))
            plt.close()

        elif type(model) in [RiskRegressionModel, RiskLapseRegressionModel]:
            print('YOOOO')

            p1, p2 = idata.posterior[pair[0]+'_mu'].to_dataframe(), idata.posterior[pair[1]+'_mu'].to_dataframe()

            for regressor in p1.index.unique(level=-1):
                if regressor in p2.index.unique(level=-1):

                    d = pd.concat((p1.xs(regressor, 0, -1), p2.xs(regressor, 0, -1)), keys=pair, names=['Variable']).stack().to_frame('Value')

                    if (name in ['prior_std', 'risky_prior_std', 'safe_prior_std', 'n1_evidence_sd', 'n2_evidence_sd', 'evidence_sd']) & (regressor == 'Intercept'):
                        d = softplus(d)

                    if (name in ['p_lapse']) & (regressor == 'Intercept'):
                        d = logistic(d)

                    g = sns.FacetGrid(d.reset_index(), hue='Variable', palette=palette)
                    g.map(sns.kdeplot, 'Value', fill=True)
                    g.add_legend()

                    g.fig.suptitle(session)
                    plt.savefig(op.join(target_folder, f'group_par-{name}.{regressor}.pdf'))
                    plt.savefig(op.join(target_folder, f'group_par-{name}.{regressor}.png'))
                    plt.close()

def plot_ppcs(model_label, ppc, df, bids_folder, session, col_wrap, group_only):

    levels = ['group']
    if not group_only:
        levels += ['subject']

    plot_types = [0, 1,2,3, 5]

    if model_label.startswith('pupil'):
        plot_types += [13, 14, 15]

    if model_label.startswith('subcortical'):
        plot_types += [17]

    if model_label.startswith('uncertainty'):
        plot_types += [6, 7, 8, 9, 10]

    if model_label.startswith('neural') or model_label.startswith('rnp_neural'):
        plot_types += [11, 12, 16]

    for plot_type in plot_types:
        for var_name in ['p', 'll_bernoulli']:
            for level in ['subject', 'group']:
                if session:
                    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'figures', model_label, session, level, var_name)
                else:
                    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'figures', model_label, level, var_name)

                if not op.exists(target_folder):
                    os.makedirs(target_folder)

                g = plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name, col_wrap=col_wrap)
                g.fig.suptitle(session)
                fn = f'{level}_plot-{plot_type}_model-{model_label}_pred.pdf'
                g.savefig(op.join(target_folder, fn))
                fn = f'{level}_plot-{plot_type}_model-{model_label}_pred.png'
                g.savefig(op.join(target_folder, fn))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('session', nargs='?', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    parser.add_argument('--no_trace', dest='plot_traces', action='store_false')
    parser.add_argument('--only_ppc', dest='only_ppc', action='store_true')
    parser.add_argument('--group_only', dest='group_only', action='store_true')
    parser.add_argument('--roi', default=None)
    args = parser.parse_args()

    main(args.model_label, args.session, bids_folder=args.bids_folder, plot_traces=args.plot_traces, group_only=args.group_only,
    only_ppc=args.only_ppc, roi=args.roi)