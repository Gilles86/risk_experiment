"""Figure 6 -- symbolic (Arabic-numeral) experiment psychophysics (panel B).

New Figure 6 of the NCOMMS-24-63995B revision is the symbolic experiment
(it was Figure 5 in the previous numbering). Its single data-driven panel is
regenerated here:

  * **6B** -- psychophysical curves per stake size (the five ``n_safe`` bins
    ``5-7`` / ``7-9`` / ``9-14`` / ``14-19`` / ``19-28``): proportion of risky
    choices vs the (binned) risky/safe payoff ratio, hue = Order (Safe first /
    Risky first), laid out to mirror Figure 1C -- two stacked rows per stake bin:
    the psychophysical curve on top, and directly below it (same width) the key
    result, the risk-neutral-point (RNP) *difference* (Safe first − Risky first).
    The points + curves are the posterior-predictive of the probit model
    (model 2: ``log_risky_safe * C(n_safe_bin) * order``); the RNP difference is
    derived per posterior draw from the group-level RNP posterior of the same
    fit.

Panel 6A (symbolic trial-sequence schematic) is hand-drawn and is NOT
regenerated here -- it is a manual Affinity ingredient.

The expensive step -- reading the probit trace and pushing posterior-predictive
samples / group-level RNP through the bambi model -- is cached to source-data
TSVs, so re-styling never re-runs it. Pass --recompute to force it. Nothing here
changes any number: the plotted quantities reproduce ``ppcs_symbolic.ipynb``
exactly (its ``invprobit`` / ``get_fake_data`` / ``extract_intercept_gamma`` /
``get_info`` logic is reused verbatim).

Usage
-----
    python -m risk_experiment.figures.figure_06_symbolic
    python -m risk_experiment.figures.figure_06_symbolic --recompute
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

from risk_experiment.figures import style


PANEL_B = 'figure_06_symbolic_psychophysics'

MODEL_LABEL = 2
BIDS_SYMBOLIC = '/data/ds-symbolicrisk'
TRACE = op.join(BIDS_SYMBOLIC, 'derivatives', 'risk_model', 'psychophysical',
                f'model{MODEL_LABEL}_samples.nc')

HUE_ORDER = ['Safe first', 'Risky first']
# Stake bins in the experiment's natural (ascending) order.
STAKE_ORDER = ['5-7', '7-9', '9-14', '14-19', '19-28']

# Risky/safe ratio binning for the curves (matches the notebook).
N_RATIO_BINS = 9
RNP_REFERENCE = 0.55  # the symbolic gamble's true risk-neutral point (55%).


# ---------------------------------------------------------------------------
# Notebook logic, reused verbatim (only formatting differs downstream)
# ---------------------------------------------------------------------------
def invprobit(x):
    return ss.norm.ppf(x)


def get_fake_data(data, group, model_label=None):
    unique_subjects = data.index.unique(level='subject')

    if group:
        fake_data = pd.MultiIndex.from_product(
            [unique_subjects[:1], [0, 1], data['n_safe_bin'].unique(),
             ['Risky first', 'Safe first']],
            names=['subject', 'log_risky_safe', 'n_safe_bin', 'order']
        ).to_frame(index=False)
    else:
        fake_data = pd.MultiIndex.from_product(
            [unique_subjects, [0, 1], data['n_safe_bin'].unique(),
             ['Risky first', 'Safe first']],
            names=['subject', 'log_risky_safe', 'n_safe_bin', 'order']
        ).to_frame(index=False)

    if model_label == 4:
        fake_data['n_safe'] = fake_data['n_safe_bin'].apply(
            lambda x: np.mean(np.array(x.split('-'), dtype=float)))

    return fake_data


def extract_intercept_gamma(trace, model, data, group=False, model_label=None):
    fake_data = get_fake_data(data, group, model_label=model_label)

    pred = model.predict(trace, 'response_params', fake_data, inplace=False,
                         include_group_specific=not group)['posterior']['p']

    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake_data))

    pred0 = pred.xs(0, 0, 'log_risky_safe')
    intercept = pd.DataFrame(invprobit(pred0), index=pred0.index,
                             columns=pred0.columns)
    gamma = invprobit(pred.xs(1, 0, 'log_risky_safe')) - intercept

    intercept = intercept.droplevel(0, 1)
    gamma = gamma.droplevel(0, 1)

    return intercept, gamma


# ---------------------------------------------------------------------------
# Posterior-predictive curves (panel 6B scatter + lines + HDI band)
# ---------------------------------------------------------------------------
def compute_curves(bids_folder='/data/ds-risk'):
    """Posterior-predictive prop. risky choices per (stake, Order, ratio-bin).

    Reproduces the notebook's ``tmp`` table: per-draw group means of the
    observed and predicted choice proportions, ready for scatter + line + HDI.
    """
    import pymc as pm
    style.shim_pymc_for_bauer(pm)
    from risk_experiment.symbolic_experiment.fit_probit import (
        get_data, build_model)

    df = get_data(model_label=MODEL_LABEL)
    model = build_model(model_label=MODEL_LABEL)
    idata = az.from_netcdf(TRACE)

    df['bin(risky/safe)'] = pd.cut(df['log(risky/safe)'], bins=N_RATIO_BINS)
    df['bin(risky/safe)'] = df['bin(risky/safe)'].apply(lambda x: x.mid)

    pred = model.predict(idata.sel(draw=slice(None, None, 20)),
                         data=df.reset_index(), inplace=False, kind='response')
    pred = pred.posterior_predictive['chose_risky'].to_dataframe()
    pred = pred.unstack([0, 1])
    pred.index = df.index
    pred = pred.droplevel(0, axis=1).stack([0, 1]).to_frame('chose_risky_pred')
    pred = pred.join(df)

    tmp = (pred
           .groupby(['subject', 'n_safe_bin', 'Order', 'chain', 'draw',
                     'bin(risky/safe)'], observed=True)
           [['chose_risky', 'chose_risky_pred']].mean()
           .groupby(['n_safe_bin', 'Order', 'bin(risky/safe)', 'chain', 'draw'],
                    observed=True).mean())

    return tmp.reset_index()


# ---------------------------------------------------------------------------
# Group-level RNP posteriors (the two insets)
# ---------------------------------------------------------------------------
def compute_rnp(bids_folder='/data/ds-risk'):
    """Group-level RNP posterior per (Order, stake), as ``get_info(2, group=True)``."""
    import pymc as pm
    style.shim_pymc_for_bauer(pm)
    from risk_experiment.symbolic_experiment.fit_probit import (
        get_data, build_model)

    df = get_data(model_label=MODEL_LABEL)
    model = build_model(model_label=MODEL_LABEL)
    idata = az.from_netcdf(TRACE)

    intercept, gamma = extract_intercept_gamma(idata, model, df, group=True,
                                               model_label=MODEL_LABEL)
    rnp = np.clip(np.exp(intercept / gamma), 0, 1)
    rnp = rnp.stack([0, 1]).to_frame('rnp')
    intercept = intercept.stack([0, 1]).to_frame('intercept')
    gamma = gamma.stack([0, 1]).to_frame('gamma')

    pars = pd.concat([intercept, gamma, rnp], axis=1).reset_index()
    return pars


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def load_or_compute(bids_folder='/data/ds-risk', recompute=False):
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)
    curves_tsv = op.join(sd_dir, f'{PANEL_B}_curves.tsv')
    rnp_tsv = op.join(sd_dir, f'{PANEL_B}_rnp.tsv')

    if recompute or not op.exists(curves_tsv) or not op.exists(rnp_tsv):
        curves = compute_curves(bids_folder)
        curves.to_csv(curves_tsv, sep='\t', index=False)
        print(f'Wrote source data: {curves_tsv}')
        rnp = compute_rnp(bids_folder)
        rnp.to_csv(rnp_tsv, sep='\t', index=False)
        print(f'Wrote source data: {rnp_tsv}')
    else:
        curves = pd.read_csv(curves_tsv, sep='\t')
        rnp = pd.read_csv(rnp_tsv, sep='\t')
        print(f'Loaded cached source data: {curves_tsv}, {rnp_tsv}')

    return curves, rnp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _hdi(x, hdi_prob=0.95):
    return pd.Series(az.hdi(np.asarray(x), hdi_prob=hdi_prob),
                     index=['hdi_lower', 'hdi_upper'])


def compute_rnp_difference(rnp):
    """Per-draw RNP difference (Safe first − Risky first) per stake.

    Derived from the existing group-level RNP posterior (faithful to the
    notebook): for each (stake, chain, draw) subtract the Risky-first RNP from
    the Safe-first RNP. No numbers change -- it is the same posterior, recast as
    a contrast.
    """
    tmp = rnp.set_index(['n_safe_bin', 'order', 'chain', 'draw'])['rnp']
    tmp = tmp.unstack('order')
    diff = (tmp['Safe first'] - tmp['Risky first']).to_frame('difference')
    return diff.reset_index()


def plot_B(curves, rnp, bids_folder='/data/ds-risk'):
    """Panel 6B, mirroring Figure 1C exactly.

    Two stacked rows per stake bin on a *shared* log(risky/safe) x-axis:
      * top (shorter than before): the psychophysical curve -- observed
        group-mean points + the probit posterior-predictive mean line and 95%
        HDI band, per Order.
      * bottom: the raw risk-neutral point (RNP) per Order, drawn as in 1C at
        ``-log(RNP)`` (where the probit crosses p = 0.5), with the risk-neutral
        reference (RNP = 0.55) marked. Risk-seeking left of it, risk-averse
        right. (The old single Δ-RNP marker is replaced by the two raw RNPs.)
    """
    import matplotlib.pyplot as plt

    style.set_style()

    n = len(STAKE_ORDER)
    mid = n // 2  # centre column -> carries the single small axis labels
    # Curves shorter (height_ratios as in 1C); RNP strip below.
    fig, axes = plt.subplots(2, n, figsize=(style.WIDTH_DOUBLE, 2.5),
                             sharey='row',
                             gridspec_kw={'height_ratios': [2.6, 1.0]},
                             constrained_layout=True)

    NEUTRAL = -np.log(RNP_REFERENCE)   # risk-neutral point in the shared log axis
    # Cover both the curve log-ratios (~0.08-1.31) and the -log(RNP) markers.
    XLIM = (0.05, 1.38)
    ratio_ticks = [1.5, 2, 3]          # natural-space ratios for the top axis
    rnp_ticks = [0.7, 0.55, 0.4]       # RNP values for the bottom axis

    for j, stake in enumerate(STAKE_ORDER):
        ax, rax = axes[0, j], axes[1, j]
        sub = curves[curves['n_safe_bin'] == stake]

        # --- top: psychophysical curve + probit PPC overlay ---
        ax.axhline(0.5, ls='--', c='k', lw=0.6, zorder=0)
        ax.axvline(NEUTRAL, ls='--', c='k', lw=0.6, zorder=0)
        for order in HUE_ORDER:
            osub = sub[sub['Order'] == order]
            color = style.ORDER_COLORS[order]
            obs = (osub.groupby('bin(risky/safe)', observed=True)['chose_risky']
                   .first())
            grp = osub.groupby('bin(risky/safe)', observed=True)['chose_risky_pred']
            mean = grp.mean()
            hdi = grp.apply(lambda v: _hdi(v)).unstack()
            xs = mean.index.values
            ax.fill_between(xs, hdi['hdi_lower'].values, hdi['hdi_upper'].values,
                            color=color, alpha=0.18, lw=0, zorder=1)
            ax.plot(xs, mean.values, '-', color=color, lw=1.1, zorder=2)
            ax.plot(obs.index.values, obs.values, 'o', ms=3.0, color=color,
                    mec='none', zorder=3)

        ax.set_title(f'Safe {stake}', pad=6, **style.BOLD)
        ax.set_xlim(*XLIM)
        ax.set_xticks([np.log(t) for t in ratio_ticks])
        ax.set_xticklabels([f'{t:g}' for t in ratio_ticks])
        ax.set_xlabel('Risky / safe', fontsize=7)
        ax.set_yticks([0, .5, 1.])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('P(risky choice)' if j == 0 else '')
        sns.despine(ax=ax, offset=3, trim=False)

        # --- bottom: raw RNP per Order at -log(RNP) (same x-axis as the curve) ---
        rsub = rnp[rnp['n_safe_bin'] == stake]
        rax.axvline(NEUTRAL, ls='--', c='k', lw=0.6, zorder=0)
        for k, order in enumerate(HUE_ORDER):
            vals = -np.log(rsub.loc[rsub['order'] == order, 'rnp'].values)
            m = vals.mean()
            lo, hi = az.hdi(vals)
            rax.errorbar(m, -k, xerr=[[m - lo], [hi - m]], fmt='o', ms=4,
                         color=style.ORDER_COLORS[order], elinewidth=1.4,
                         capsize=0)
        rax.set_xlim(*XLIM)
        rax.set_xticks([-np.log(t) for t in rnp_ticks])
        rax.set_xticklabels([f'{t:g}'.lstrip('0') for t in rnp_ticks])
        rax.set_ylim(-1.9, 0.9)
        rax.set_yticks([])
        rax.set_xlabel('RNP', fontsize=7)
        rax.text(XLIM[0] + 0.05, 0.8, 'Risk-seeking', fontsize=6, ha='left',
                 va='bottom', color='0.45')
        rax.text(XLIM[1] - 0.05, 0.8, 'Risk-averse', fontsize=6, ha='right',
                 va='bottom', color='0.45')
        sns.despine(ax=rax, left=True, offset=3, trim=False)
        rax.tick_params(axis='y', length=0)

    # No in-panel key -- the Order legend is a separate ingredient placed by hand
    # in Affinity so it never overlaps the small panels (matches Figure 1C).

    pdf = style.save_panel(fig, PANEL_B, bids_folder, tight=False)
    print(f'Wrote figure: {pdf}')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-folder', default='/data/ds-risk')
    parser.add_argument('--recompute', action='store_true',
                        help='Force recomputation of the posterior predictives.')
    args = parser.parse_args()

    curves, rnp = load_or_compute(args.bids_folder, recompute=args.recompute)
    plot_B(curves, rnp, args.bids_folder)
