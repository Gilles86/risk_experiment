#!/usr/bin/env python3
"""
Reproduce paper figures from Source_Data.xlsx (and some traces directly).
Run from repo root: python paper/plot_source_data.py
Output: paper/source_data_figures.pdf
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import arviz as az
import os.path as op
import scipy.stats as ss
from matplotlib.backends.backend_pdf import PdfPages
from bauer.utils.math import softplus_np

SOURCE = op.join(op.dirname(op.abspath(__file__)), 'Source_Data.xlsx')
OUTPUT = op.join(op.dirname(op.abspath(__file__)), 'source_data_figures.pdf')
BIDS_FOLDER = '/data/ds-risk'
SYMBOLIC_BIDS_FOLDER = '/data/ds-symbolicrisk'

xl = pd.ExcelFile(SOURCE)
sheets = {name: xl.parse(name) for name in xl.sheet_names}

# ── colour palette (seaborn tab10 conventions) ────────────────────────────────
_tab     = sns.color_palette('tab10')
C_BLUE   = _tab[0]   # safe / safe-first / risk-averse
C_ORANGE = _tab[1]   # risky-first
C_GREEN  = _tab[2]   # low neural uncertainty
C_RED    = _tab[3]   # risky / high neural uncertainty / risk-seeking
C_PURPLE = _tab[4]   # option 2 (n2)
C_BROWN  = _tab[5]   # option 1 (n1)
C_GRAY   = '#888888' # neutral / risk-neutral

ORDER_COLORS  = {'Risky first': C_ORANGE, 'Safe first': C_BLUE}
RISK_COLORS   = {'risk-seeking': C_RED, 'risk-neutral': C_GRAY, 'risk-averse': C_BLUE}
RISK_MARKERS  = {'risk-seeking': 'o', 'risk-neutral': 's', 'risk-averse': '^'}
UNCERT_COLORS = {'High uncertainty': C_RED, 'Low uncertainty': C_GREEN}
STAKE_BINS    = ['Small (5-7)', 'Medium (10-14)', 'Large (20-28)']


def _param_color(label):
    """Canonical colour for a PMC parameter label."""
    if '(n1)' in label or 'n1' in label.lower():
        return C_BROWN
    if '(n2)' in label or 'n2' in label.lower():
        return C_PURPLE
    if 'risky' in label.lower() or 'Risky' in label:
        return C_RED
    if 'safe' in label.lower() or 'Safe' in label:
        return C_BLUE
    return 'black'

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
                     'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'figure.dpi': 120})


def finish(ax, xlabel=None, ylabel=None, title=None):
    ax.spines[['top', 'right']].set_visible(False)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title, fontweight='bold')


def combined_session(idata, param):
    """Average 3T/7T: softplus(Intercept + 0.5 * session[T.7t2])"""
    df_p = idata.posterior[param].to_dataframe()
    intercept      = df_p.xs('Intercept',        level=-1).iloc[:, 0]
    session_effect = df_p.xs('session[T.7t2]',   level=-1).iloc[:, 0]
    return softplus_np(intercept + 0.5 * session_effect)


# ── load risk preferences (session='both') ────────────────────────────────────
risk_prefs = pd.read_csv(
    op.join(BIDS_FOLDER, 'derivatives', 'cogmodels', 'simple_risk_preference.tsv'),
    index_col=[0, 1], sep='\t', dtype={'subject': str}
)
rp_both = risk_prefs.xs('both', level='session')['risk_profile']


with PdfPages(OUTPUT) as pdf:

    # ── Figure 1B ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 3))
    df = sheets['Figure 1B']
    for order, grp in df.groupby('order'):
        grp = grp.sort_values('safe_payoff_CHF')
        ax.errorbar(grp['safe_payoff_CHF'], grp['mean'], yerr=grp['sem'],
                    color=ORDER_COLORS[order], marker='o', label=order,
                    linewidth=1.5, markersize=4, capsize=3)
    ax.axhline(0.5, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(frameon=False)
    finish(ax, xlabel='Safe payoff (CHF)', ylabel='Proportion chose risky', title='Figure 1B')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ── Figure 1C ─────────────────────────────────────────────────────────────
    df  = sheets['Figure 1C']
    rnp = sheets['Figure 1C RNP insets']
    fig, axes = plt.subplots(3, 2, figsize=(7, 7), sharey=True, sharex=True)
    for ri, stake in enumerate(STAKE_BINS):
        for ci, order in enumerate(['Risky first', 'Safe first']):
            ax  = axes[ri, ci]
            grp = df[(df['stake_bin'] == stake) & (df['order'] == order)].sort_values('log_risky_safe_ratio_bin')
            ax.errorbar(grp['log_risky_safe_ratio_bin'], grp['mean'], yerr=grp['sem'],
                        color=ORDER_COLORS[order], marker='o', markersize=3, linewidth=1.5, capsize=2)
            ax.axhline(0.5, color='k', ls='--', lw=0.7, alpha=0.5)
            r = rnp[(rnp['stake_bin'] == stake) & (rnp['order'] == order)]
            if len(r):
                ax.text(0.05, 0.92, f'RNP={r["mean"].values[0]:.2f}',
                        transform=ax.transAxes, fontsize=7, color=ORDER_COLORS[order])
            if ri == 0: ax.set_title(order)
            if ci == 0: ax.set_ylabel(f'{stake}\nProp. risky')
            finish(ax)
    axes[-1, 0].set_xlabel('log(risky/safe)'); axes[-1, 1].set_xlabel('log(risky/safe)')
    fig.suptitle('Figure 1C', fontweight='bold')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ── Figure 2A — 3 panels (stake), both orders overlaid ───────────────────
    df  = sheets['Figure 2A']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ci, stake in enumerate(STAKE_BINS):
        ax = axes[ci]
        for order in ['Safe first', 'Risky first']:
            grp   = df[(df['stake_bin'] == stake) & (df['order'] == order)].sort_values('bin(risky/safe)')
            color = ORDER_COLORS[order]
            ax.fill_between(grp['bin(risky/safe)'], grp['model_hdi_2_5'], grp['model_hdi_97_5'],
                            color=color, alpha=0.2, linewidth=0)
            ax.plot(grp['bin(risky/safe)'], grp['model_mean'], color=color, lw=1.5,
                    label=order if ci == 0 else None)
            ax.errorbar(grp['bin(risky/safe)'], grp['empirical_mean'], yerr=grp['empirical_sem'],
                        fmt='o', color=color, markersize=3, capsize=2)
        ax.axhline(0.5, color='k', ls='--', lw=0.7, alpha=0.5)
        ax.set_title(stake)
        finish(ax, xlabel='log(risky/safe) bin')
    axes[0].set_ylabel('Proportion chose risky')
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle('Figure 2A — PMC model PPC', fontweight='bold')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ── Figure 2B — KDE per parameter group ──────────────────────────────────
    print("Loading model-12 trace for Fig 2B/2C…")
    idata_12 = az.from_netcdf(op.join(BIDS_FOLDER, 'derivatives', 'cogmodels',
                                      'model-12_trace.netcdf'))

    param_groups = {
        'Evidence noise (SD)': [
            ('n1_evidence_sd_mu', '1st option (n1)'),
            ('n2_evidence_sd_mu', '2nd option (n2)'),
        ],
        'Prior mean': [
            ('risky_prior_mu_mu', 'Risky'),
            ('safe_prior_mu_mu',  'Safe'),
        ],
        'Prior width (SD)': [
            ('risky_prior_sd_mu', 'Risky'),
            ('safe_prior_sd_mu',  'Safe'),
        ],
    }

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, (group_name, params) in zip(axes, param_groups.items()):
        for param_key, param_label in params:
            vals = combined_session(idata_12, param_key).values
            sns.kdeplot(vals, ax=ax, label=param_label, color=_param_color(param_label),
                        fill=True, alpha=0.4)
        ax.set_xlabel('Parameter value'); ax.set_ylabel('Density')
        ax.legend(frameon=False, fontsize=7)
        finish(ax, title=group_name)
    fig.suptitle('Figure 2B — Group-level parameters (combined sessions)', fontweight='bold')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ── Figure 2C — marker plot, coloured by risk preference ─────────────────
    df2c = sheets['Figure 2C'].copy()
    df2c['subject'] = df2c['subject'].astype(str).str.zfill(2)
    df2c = df2c.merge(rp_both.reset_index().rename(columns={'index': 'subject'}),
                      on='subject', how='left')

    comparisons = df2c['comparison'].unique()
    fig, axes = plt.subplots(1, len(comparisons), figsize=(11, 5))
    for ax, comp in zip(axes, comparisons):
        grp = df2c[df2c['comparison'] == comp].sort_values('posterior_mean')
        for i, (_, row) in enumerate(grp.iterrows()):
            rp     = row.get('risk_profile', 'risk-neutral')
            color  = RISK_COLORS.get(rp, 'gray')
            marker = RISK_MARKERS.get(rp, 'o')
            ax.plot(row['posterior_mean'], i, marker=marker, color=color, markersize=6, zorder=5)
            ax.hlines(i, row['hdi_2_5'], row['hdi_97_5'], color=color, lw=1.2, alpha=0.7)
        ax.axvline(0, color='k', lw=1)
        ax.set_yticks(range(len(grp)))
        ax.set_yticklabels([f"sub-{s}" for s in grp['subject']], fontsize=5)
        finish(ax, title=grp['description'].values[0], xlabel='Posterior difference')

    legend_els = [
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor=RISK_COLORS['risk-seeking'],  markersize=7, label='Risk-seeking'),
        mlines.Line2D([], [], marker='s', color='w', markerfacecolor=RISK_COLORS['risk-neutral'],  markersize=7, label='Risk-neutral'),
        mlines.Line2D([], [], marker='^', color='w', markerfacecolor=RISK_COLORS['risk-averse'],   markersize=7, label='Risk-averse'),
    ]
    axes[-1].legend(handles=legend_els, frameon=False, fontsize=7, loc='lower right')
    fig.suptitle('Figure 2C — Subject-level parameter differences', fontweight='bold')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ── Figure 3A-D — all 3 stake sizes per model ────────────────────────────
    df = sheets['Figure 3A-D']
    models = df['model'].unique()
    fig, axes = plt.subplots(len(models), 3, figsize=(11, 2.8 * len(models)),
                             sharey=True, sharex=True)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]
    for mi, model in enumerate(models):
        mdf = df[df['model'] == model]
        for ci, stake in enumerate(STAKE_BINS):
            ax = axes[mi, ci]
            for order in ['Safe first', 'Risky first']:
                grp   = mdf[(mdf['stake_bin'] == stake) & (mdf['order'] == order)].sort_values('bin(risky/safe)')
                color = ORDER_COLORS[order]
                ax.fill_between(grp['bin(risky/safe)'], grp['model_hdi_2_5'], grp['model_hdi_97_5'],
                                color=color, alpha=0.2, linewidth=0)
                ax.plot(grp['bin(risky/safe)'], grp['model_mean'], color=color, lw=1.5)
                ax.errorbar(grp['bin(risky/safe)'], grp['empirical_mean'], yerr=grp['empirical_sem'],
                            fmt='o', color=color, markersize=2.5, capsize=2)
            ax.axhline(0.5, color='k', ls='--', lw=0.7, alpha=0.5)
            if mi == 0: ax.set_title(stake, fontsize=8)
            if ci == 0: ax.set_ylabel(model[:22] + '\nProp. risky', fontsize=7)
            finish(ax)
    for ci in range(3):
        axes[-1, ci].set_xlabel('log(risky/safe) bin')
    fig.suptitle('Figure 3A-D — Alternative model PPCs', fontweight='bold')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ── Figure 3E ─────────────────────────────────────────────────────────────
    df = sheets['Figure 3E'].sort_values('elpd_loo', ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    y = np.arange(len(df))
    ax.barh(y, df['elpd_loo'] - df['elpd_loo'].max(), xerr=df['dse'],
            color='steelblue', alpha=0.8, capsize=3, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(df['model'], fontsize=8)
    ax.axvline(0, color='k', lw=0.8)
    finish(ax, xlabel='ΔELPD-LOO (vs best model)', title='Figure 3E — Model comparison')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ── Figure 4B — swarm plot ────────────────────────────────────────────────
    df = sheets['Figure 4B']
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for ax, col, ylabel in zip(
        axes,
        ['r_decoded_vs_actual_numerosity', 'r_decoded_uncertainty_vs_abs_error'],
        ['r (decoded vs actual)', 'r (uncertainty vs |error|)'],
    ):
        sns.swarmplot(data=df, x='scanner', y=col, ax=ax,
                      hue='scanner', palette={'3T': '#5C85D6', '7T': '#D65C5C'},
                      legend=False, size=4)
        ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        finish(ax, xlabel='Scanner', ylabel=ylabel)
    fig.suptitle('Figure 4B — Decoding accuracy', fontweight='bold')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ── Figure 4C — KDE from full posterior ──────────────────────────────────
    print("Loading probit_neural9 trace for Fig 4C…")
    try:
        from risk_experiment.cogmodels.fit_probit import (
            build_model as build_probit, get_data as get_probit_data)
        from risk_experiment.cogmodels.utils import extract_intercept_gamma

        df_n    = get_probit_data(model_label='probit_neural9', session=None, bids_folder=BIDS_FOLDER)
        model_n = build_probit(model_label='probit_neural9', df=df_n, session=None, bids_folder=BIDS_FOLDER)
        idata_n = az.from_netcdf(op.join(BIDS_FOLDER, 'derivatives', 'cogmodels',
                                         'model-probit_neural9_trace.netcdf'))

        intercept, gamma = extract_intercept_gamma(idata_n, model_n, df_n, group=False)
        rnp = np.exp(intercept['intercept'] / gamma['gamma']).stack([-2, -1]).to_frame('rnp')
        rnp = rnp[(rnp['rnp'] > 0.0) & (rnp['rnp'] < 1.0)]
        rnp['distance'] = (rnp['rnp'] - 0.55).abs()

        fig, ax = plt.subplots(figsize=(4, 3))
        for neural_unc, label in [
            (True,  'High uncertainty'),
            (False, 'Low uncertainty'),
        ]:
            color = UNCERT_COLORS[label]
            vals = (rnp.xs(neural_unc, level='median_split_sd')['distance']
                       .groupby(['chain', 'draw']).mean().values)
            sns.kdeplot(vals, ax=ax, label=label, color=color, fill=True, alpha=0.4)
        ax.set_xlabel('Mean distance to risk neutrality')
        ax.legend(frameon=False)
        finish(ax, ylabel='Density', title='Figure 4C')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    except Exception as e:
        print(f"  Fig 4C KDE failed: {e}")

    # ── Figure 4D — KDE of slopes per parameter group ────────────────────────
    print("Loading neural32 trace for Fig 4D…")
    try:
        idata_n32 = az.from_netcdf(op.join(BIDS_FOLDER, 'derivatives', 'cogmodels',
                                           'model-neural32_trace.netcdf'))

        param_groups_4d = {
            'Evidence noise (SD)': [
                ('n1_evidence_sd_mu', '1st option (n1)'),
                ('n2_evidence_sd_mu', '2nd option (n2)'),
            ],
            'Prior mean': [
                ('risky_prior_mu_mu', 'Risky'),
                ('safe_prior_mu_mu',  'Safe'),
            ],
            'Prior width (SD)': [
                ('risky_prior_sd_mu', 'Risky'),
                ('safe_prior_sd_mu',  'Safe'),
            ],
        }

        def session_avg_slope(pvar):
            """Average sd slope across sessions: sd + 0.5 * sd:session[T.7t2]"""
            regdim = [d for d in pvar.dims if d not in ('chain', 'draw')][0]
            coords = list(pvar.coords[regdim].values)
            sd_main = pvar.sel({regdim: 'sd'}).values.ravel()
            inter_key = next((c for c in coords if 'sd:session' in c or 'sd_session' in c
                              or ('sd' in c and '7t2' in c)), None)
            if inter_key:
                sd_inter = pvar.sel({regdim: inter_key}).values.ravel()
                return sd_main + 0.5 * sd_inter
            return sd_main   # fallback: no interaction term found

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, (group_name, params) in zip(axes, param_groups_4d.items()):
            for param_key, param_label in params:
                if param_key not in idata_n32.posterior:
                    continue
                pvar   = idata_n32.posterior[param_key]
                regdim = [d for d in pvar.dims if d not in ('chain', 'draw')]
                if regdim and 'sd' in list(pvar.coords[regdim[0]].values):
                    slope_vals = session_avg_slope(pvar)
                    sns.kdeplot(slope_vals, ax=ax, label=param_label,
                                color=_param_color(param_label), fill=True, alpha=0.4)
            ax.axvline(0, color='k', ls='--', lw=1)
            ax.set_xlabel('Slope (decoded neural SD, session-averaged)')
            ax.legend(frameon=False, fontsize=7)
            finish(ax, ylabel='Density', title=group_name)
        fig.suptitle('Figure 4D — Neural uncertainty slopes on PMC parameters (session-averaged)', fontweight='bold')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    except Exception as e:
        print(f"  Fig 4D KDE failed: {e}")

    # ── Figure 5B — overlay orders, axvline at risk-neutrality ───────────────
    df = sheets['Figure 5B']
    stake_bins_sym = sorted(df['stake_bin'].unique(), key=lambda x: float(x.split('-')[0]))
    risk_neutral_x = np.log(1. / 0.55)

    n_stakes = len(stake_bins_sym)
    fig, axes = plt.subplots(1, n_stakes, figsize=(3 * n_stakes, 3.5), sharey=True)
    if n_stakes == 1:
        axes = [axes]

    for ci, stake in enumerate(stake_bins_sym):
        ax = axes[ci]
        for order in ['Safe first', 'Risky first']:
            grp   = df[(df['stake_bin'] == stake) & (df['order'] == order)].sort_values('log_risky_safe_ratio_bin')
            color = ORDER_COLORS[order]
            ax.errorbar(grp['log_risky_safe_ratio_bin'], grp['mean'], yerr=grp['sem'],
                        color=color, marker='o', markersize=3, lw=1.5, capsize=2,
                        label=order if ci == 0 else None)
        ax.axhline(0.5, color='k', ls='--', lw=0.7, alpha=0.5)
        ax.axvline(risk_neutral_x, color='gray', ls=':', lw=1)
        ax.set_title(f'Safe payoff: {stake}')
        finish(ax, xlabel='log(risky/safe)')

    axes[0].set_ylabel('Proportion chose risky')
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle('Figure 5B — Symbolic experiment', fontweight='bold')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ── Figure 5B RNP — indifference ratios and order differences ────────────
    print("Loading symbolic trace for Fig 5B RNP…")
    try:
        from risk_experiment.symbolic_experiment.fit_probit import (
            build_model as build_sym, get_data as get_sym_data)

        model_label_sym = 2
        df_sym   = get_sym_data()
        model_sym = build_sym(model_label=model_label_sym)
        idata_sym = az.from_netcdf(op.join(SYMBOLIC_BIDS_FOLDER, 'derivatives',
                                           'risk_model', 'psychophysical',
                                           f'model{model_label_sym}_samples.nc'))

        def invprobit(x):
            return ss.norm.ppf(x)

        # Group-level predictions (replicate notebook's extract_intercept_gamma)
        unique_subjects = df_sym.index.unique(level='subject')
        fake_data = pd.MultiIndex.from_product(
            [unique_subjects[:1], [0, 1], df_sym['n_safe_bin'].unique(),
             ['Risky first', 'Safe first']],
            names=['subject', 'log_risky_safe', 'n_safe_bin', 'order']
        ).to_frame(index=False)

        pred = model_sym.predict(idata_sym, 'response_params', fake_data,
                                  inplace=False, include_group_specific=False)['posterior']['p']
        pred = pred.to_dataframe().unstack([0, 1])
        pred = pred.set_index(pd.MultiIndex.from_frame(fake_data))

        pred0     = pred.xs(0, 0, 'log_risky_safe')
        intercept = pd.DataFrame(invprobit(pred0),   index=pred0.index, columns=pred0.columns)
        gamma     = invprobit(pred.xs(1, 0, 'log_risky_safe')) - intercept
        intercept = intercept.droplevel(0, 1)
        gamma     = gamma.droplevel(0, 1)

        rnp_sym = np.clip(np.exp(intercept / gamma), 0, 1)
        rnp_sym = rnp_sym.stack([0, 1]).to_frame('rnp')

        indiff = (1. / rnp_sym['rnp']).to_frame('indifference_ratio')

        # -- plot 1: catplot-style indifference ratio by stake and order -------
        fig, axes = plt.subplots(1, n_stakes, figsize=(3 * n_stakes, 3.5))
        if n_stakes == 1:
            axes = [axes]

        for ci, stake in enumerate(stake_bins_sym):
            ax = axes[ci]
            for oi, (order, color) in enumerate(
                [('Safe first', ORDER_COLORS['Safe first']),
                 ('Risky first', ORDER_COLORS['Risky first'])]
            ):
                try:
                    vals   = indiff.xs(stake, level='n_safe_bin').xs(order, level='order')['indifference_ratio'].values
                    mean_v = vals.mean()
                    hdi95  = az.hdi(vals, hdi_prob=0.95)
                    ax.scatter(mean_v, oi, color=color, s=60, zorder=5,
                               label=order if ci == 0 else None)
                    ax.hlines(oi, hdi95[0], hdi95[1], color=color, lw=2)
                except Exception:
                    pass
            ax.axvline(1. / 0.55, color='k', ls='--', lw=0.8)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Safe first', 'Risky first'], fontsize=7)
            ax.set_title(f'Safe payoff: {stake}')
            finish(ax, xlabel='Indifference ratio (risky/safe)')

        axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle('Figure 5B RNP — Indifference ratios by stake and order', fontweight='bold')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # -- plot 2: KDE of Risky first − Safe first difference ---------------
        diff_df = indiff.unstack('order').droplevel(0, 1)
        diff_df = (diff_df['Risky first'] - diff_df['Safe first']).to_frame('Indifference point difference')

        fig, axes = plt.subplots(1, n_stakes, figsize=(3 * n_stakes, 3), sharey=True)
        if n_stakes == 1:
            axes = [axes]

        for ci, stake in enumerate(stake_bins_sym):
            ax = axes[ci]
            try:
                vals = diff_df.xs(stake, level='n_safe_bin')['Indifference point difference'].values
                sns.kdeplot(vals, ax=ax, color='k', fill=True, alpha=0.4)
                ax.axvline(0, color='k', ls='--', lw=0.8)
            except Exception:
                pass
            ax.set_title(f'Safe payoff: {stake}')
            finish(ax, xlabel='Risky first − Safe first (indiff. ratio)')
        axes[0].set_ylabel('Density')
        fig.suptitle('Figure 5B RNP — Order difference in indifference ratio', fontweight='bold')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    except Exception as e:
        print(f"  Fig 5B RNP failed: {e}")

print(f"\nSaved to {OUTPUT}")
