"""Figure 2 -- the PMCM model: posterior predictions (A), group-level parameter
posteriors (B), and participant-level parameter differences (C).

Per the manuscript, Figure 2 combines the 3T and 7T sessions: it uses the
combined model ``model-12``. That model carries a ``session`` regressor, so the
"combined" estimate of each parameter is the mean across the two sessions --
exactly as the original figure2.ipynb computes it:

    softplus_np(x.xs('Intercept') + 0.5 * x.xs('session[T.7t2]'))

(intercept = 3T level; intercept + offset = 7T level; their mean = intercept +
0.5*offset, then the softplus link). Reproduced verbatim here so the numbers are
unchanged. The 2C participant colours use the *combined* risk profile
(``simple_risk_preference.tsv`` session == 'both').

Panels (per-panel vector PDF ingredients, assembled in Affinity):
  2A  figure_02A_ppc                 -- PMCM posterior-predictive curves
  2B  figure_02B_group_posteriors    -- group-level posteriors (3 KDEs)
  2C  figure_02C_participant_diffs   -- participant-level differences (3 forests)

Usage:
    python -m risk_experiment.figures.figure_02_pmcm
    python -m risk_experiment.figures.figure_02_pmcm --recompute
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns

from risk_experiment.figures import style
from risk_experiment.figures.figure_03_alt_models import compute_summary as ppc_summary

BIDS = '/data/ds-risk'
MODEL = '12'  # combined PMCM
ORDER_LABELS = {False: 'Safe first', True: 'Risky first'}
STAKE_ORDER = ['Small', 'Medium', 'Large']

# Risk-category colours, exactly as figure2.ipynb.
RISK_COLORS = {'risk-neutral': 'k',
               'risk-seeking': sns.color_palette('Spectral', 5)[0],
               'risk-averse': sns.color_palette('Spectral', 5)[-1]}
RISK_LABELS = {'risk-seeking': 'Risk-seeking', 'risk-averse': 'Risk-averse',
               'risk-neutral': 'Risk-neutral'}


def _hdi(x, prob=0.95):
    lo, hi = az.hdi(np.asarray(x), hdi_prob=prob)
    return pd.Series({'hdi_lower': lo, 'hdi_upper': hi})


def _combine_sessions(idata, var):
    """Combined (session-mean) value of a model-12 parameter, softplus link.

    Returns a DataFrame (column == var) indexed by everything except the
    regressor dimension. Verbatim from the original figure2.ipynb idiom.
    """
    from bauer.utils.math import softplus_np
    df = idata.posterior[var].to_dataframe()
    return softplus_np(df.xs('Intercept', 0, -1)
                       + 0.5 * df.xs('session[T.7t2]', 0, -1))


# --------------------------------------------------------------------------- #
# 2B -- group-level posteriors
# --------------------------------------------------------------------------- #
GROUP_SPECS = [
    # (title, var_risky/n1, var_safe/n2, label_a, label_b, hue_order, palette)
    ('Evidence SD', 'n1_evidence_sd_mu', 'n2_evidence_sd_mu', 'Option 1', 'Option 2',
     ['Option 1', 'Option 2'], list(sns.color_palette('tab10')[4:6])),
    ('Prior mu', 'risky_prior_mu_mu', 'safe_prior_mu_mu', 'Risky', 'Safe',
     ['Safe', 'Risky'], [sns.color_palette('coolwarm', 4)[i] for i in (0, 3)]),
    ('Prior std', 'risky_prior_sd_mu', 'safe_prior_sd_mu', 'Risky', 'Safe',
     ['Safe', 'Risky'], [sns.color_palette('coolwarm', 4)[i] for i in (0, 3)]),
]


def compute_group(idata):
    frames = []
    for title, va, vb, la, lb, _, _ in GROUP_SPECS:
        a = _combine_sessions(idata, va).iloc[:, 0].reset_index(drop=True)
        b = _combine_sessions(idata, vb).iloc[:, 0].reset_index(drop=True)
        frames.append(pd.DataFrame({'panel': title, la: a, lb: b})
                      .melt(id_vars='panel', var_name='condition',
                            value_name='value'))
    return pd.concat(frames, ignore_index=True)


def plot_group(group, bids_folder=BIDS):
    import matplotlib.pyplot as plt
    style.set_style()
    fig, axes = style.figure2_row(3)                # shared aligned geometry
    for ax, (title, va, vb, la, lb, hue_order, palette) in zip(axes, GROUP_SPECS):
        sub = group[group['panel'] == title]
        cmap = dict(zip(hue_order, palette))
        for cond in hue_order:
            vals = sub[sub['condition'] == cond]['value'].values
            sns.kdeplot(x=vals, fill=True, color=cmap[cond], alpha=0.5,
                        lw=1.0, ax=ax, label=cond)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('Posterior density' if ax is axes[0] else '')
        ax.set_yticks([])
        leg = ax.legend(frameon=True, fontsize=8, handlelength=1.0,
                        handletextpad=0.4, edgecolor='0.6', facecolor='white',
                        framealpha=1.0)
        leg.get_frame().set_linewidth(0.6)
    # Keep the y-axis spine to anchor the densities (no numeric ticks -- KDE
    # height is arbitrary).
    sns.despine(fig=fig, offset=3, trim=False)
    for ax in axes:
        ax.tick_params(axis='y', length=0)
    pdf = style.save_panel(fig, 'figure_02B_group_posteriors', bids_folder, tight=False)
    print(f'Wrote figure: {pdf}')


# --------------------------------------------------------------------------- #
# 2C -- participant-level differences
# --------------------------------------------------------------------------- #
DIFF_SPECS = [
    # (title, x-axis label, var_a, var_b)  -> diff = combine(a) - combine(b).
    # Titles name the parameter (as in the published panel); the x-label names
    # the difference being plotted.
    ('Evidence SD (difference)', 'Option 1 - option 2',
     'n1_evidence_sd', 'n2_evidence_sd'),
    ('Prior mu (difference)', 'Risky - safe', 'risky_prior_mu', 'safe_prior_mu'),
    ('Prior std (difference)', 'Risky - safe', 'risky_prior_sd', 'safe_prior_sd'),
]


def compute_diffs(idata, bids_folder=BIDS):
    rp = pd.read_csv(op.join(bids_folder, 'derivatives', 'cogmodels',
                             'simple_risk_preference.tsv'),
                     index_col=[0, 1], sep='\t', dtype={'subject': str})
    rp_both = rp.xs('both', 0, 'session')  # combined risk profile for colours

    out = []
    for title, _xlabel, va, vb in DIFF_SPECS:
        a = _combine_sessions(idata, va)
        b = _combine_sessions(idata, vb)
        diff = (a[va] - b[vb]).to_frame('diff')
        hdi = (diff.groupby('subject')['diff']
               .apply(lambda d: pd.Series(az.hdi(d.values),
                                          index=['low', 'high'])).unstack())
        mean = diff.groupby('subject').mean().rename(columns={'diff': 'mean'})
        tmp = mean.join(hdi).join(rp_both)
        tmp['panel'] = title
        out.append(tmp.reset_index())
    return pd.concat(out, ignore_index=True)


def plot_diffs(diffs, bids_folder=BIDS):
    import matplotlib.pyplot as plt
    style.set_style()
    fig, axes = style.figure2_row(3)                # shared aligned geometry
    for ax, (title, xlabel, _, _) in zip(axes, DIFF_SPECS):
        sub = diffs[diffs['panel'] == title].copy()
        order = sub.sort_values('mean')['subject'].tolist()
        ax.axvline(0, c='k', ls='--', lw=0.6, zorder=0)
        for _, row in sub.iterrows():
            y = -order.index(row['subject'])
            # Thicker gray HDI bars, as in the published panel.
            ax.plot([row['low'], row['high']], [y, y], color='gray', lw=2.0,
                    zorder=2)
            ax.scatter([row['mean']], [y], s=16,
                       color=RISK_COLORS[row['risk_profile']], zorder=5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.set_ylabel('Participant' if ax is axes[0] else '')
    # Keep the y-axis spine (anchors the participant rows); no in-panel legend
    # -- the risk-category key is a separate ingredient (make_risk_legend).
    sns.despine(fig=fig, offset=3, trim=False)
    for ax in axes:
        ax.tick_params(axis='y', length=0)
    pdf = style.save_panel(fig, 'figure_02C_participant_diffs', bids_folder, tight=False)
    print(f'Wrote figure: {pdf}')


def make_risk_legend(bids_folder=BIDS):
    """Standalone risk-category key (separate ingredient, as in the original)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style.set_style()
    fig, ax = plt.subplots(figsize=(34 * style.MM, 16 * style.MM))
    ax.axis('off')
    keys = ['risk-seeking', 'risk-averse', 'risk-neutral']
    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=5,
                      color=RISK_COLORS[k], label=RISK_LABELS[k]) for k in keys]
    leg = ax.legend(handles=handles, loc='center', frameon=True, fontsize=7,
                    handletextpad=0.3, labelspacing=0.4, borderpad=0.6,
                    edgecolor='0.6', facecolor='white', framealpha=1.0)
    leg.get_frame().set_linewidth(0.6)
    pdf = style.save_panel(fig, 'figure_02C_risk_legend', bids_folder,
                           tight=True, transparent=True, pad=0.05)
    print(f'Wrote figure: {pdf}')


# --------------------------------------------------------------------------- #
# 2A -- PMCM posterior-predictive curves (same machinery as Figure 3, coloured)
# --------------------------------------------------------------------------- #
def plot_ppc(summary, bids_folder=BIDS):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style.set_style()
    summary = summary.copy()
    summary['bin(n_safe)'] = pd.Categorical(summary['bin(n_safe)'],
                                            categories=STAKE_ORDER, ordered=True)
    xpos, xlab = style.ratio_bin_ticks(bids_folder)  # natural-space ratio ticks
    x_neutral = style.ratio_pos(1 / 0.55, bids_folder)  # risk-neutral ratio
    fig, axes = style.figure2_row(3, sharey=True)   # shared aligned geometry
    for ax, stake in zip(axes, STAKE_ORDER):
        sub = summary[summary['bin(n_safe)'] == stake]
        ax.axhline(0.5, c='k', ls='--', lw=0.5, zorder=0)
        ax.axvline(x_neutral, c='k', ls='--', lw=0.5, zorder=0)
        for risky_first in [False, True]:
            cell = sub[sub['risky_first'] == risky_first].sort_values('bin(risky/safe)')
            if cell.empty:
                continue
            color = style.ORDER_COLORS[ORDER_LABELS[risky_first]]
            x = cell['bin(risky/safe)'].values
            ax.fill_between(x, cell['hdi_lower'], cell['hdi_upper'],
                            color=color, alpha=0.15, lw=0, zorder=1)
            ax.plot(x, cell['pred_mean'], color=color, lw=1.1, zorder=2)
            ax.plot(x, cell['chose_risky'], marker='o', ls='', ms=3.5,
                    color=color, mec='none', zorder=3)
        ax.set_title(f'Stake size {stake}')
        ax.set_xlabel('Risky / safe payoff')
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlab)
        ax.set_yticks([0, .25, .5, .75, 1.])
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel('P(risky choice)')
    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=5,
                      color=style.ORDER_COLORS[ORDER_LABELS[k]])
               for k in [False, True]]
    # Boxed key, harmonised with the Fig 1 order legend / Fig 4. Placed lower-
    # right, where the rising curves leave space and it clears the risk-neutral
    # axvline (which sits in the left half of the panel).
    leg = axes[0].legend(handles, [ORDER_LABELS[False], ORDER_LABELS[True]],
                         loc='lower right', frameon=True, fontsize=8,
                         handletextpad=0.3, borderaxespad=0.4,
                         edgecolor='0.6', facecolor='white', framealpha=1.0)
    leg.get_frame().set_linewidth(0.6)
    sns.despine(fig=fig, offset=3, trim=False)
    pdf = style.save_panel(fig, 'figure_02A_ppc', bids_folder, tight=False)
    print(f'Wrote figure: {pdf}')


# --------------------------------------------------------------------------- #
def main(bids_folder=BIDS, recompute=False):
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)

    # ---- 2A: posterior predictive (cached like Figure 3) ----
    ppc_tsv = op.join(sd_dir, 'figure_02A_ppc.tsv')
    if recompute or not op.exists(ppc_tsv):
        ppc = ppc_summary(MODEL, bids_folder)
        ppc.to_csv(ppc_tsv, sep='\t', index=False)
        print(f'Wrote source data: {ppc_tsv}')
    else:
        ppc = pd.read_csv(ppc_tsv, sep='\t')
    plot_ppc(ppc, bids_folder)

    # ---- 2B / 2C: group + participant posteriors from model-12 ----
    idata = az.from_netcdf(op.join(bids_folder, 'derivatives', 'cogmodels',
                                   f'model-{MODEL}_trace.netcdf'))

    group_tsv = op.join(sd_dir, 'figure_02B_group_posteriors.tsv')
    if recompute or not op.exists(group_tsv):
        group = compute_group(idata)
        group.to_csv(group_tsv, sep='\t', index=False)
        print(f'Wrote source data: {group_tsv}')
    else:
        group = pd.read_csv(group_tsv, sep='\t')
    plot_group(group, bids_folder)

    diff_tsv = op.join(sd_dir, 'figure_02C_participant_diffs.tsv')
    if recompute or not op.exists(diff_tsv):
        diffs = compute_diffs(idata, bids_folder)
        diffs.to_csv(diff_tsv, sep='\t', index=False)
        print(f'Wrote source data: {diff_tsv}')
    else:
        diffs = pd.read_csv(diff_tsv, sep='\t', dtype={'subject': str})
    plot_diffs(diffs, bids_folder)
    make_risk_legend(bids_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-folder', default=BIDS)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    main(args.bids_folder, recompute=args.recompute)
