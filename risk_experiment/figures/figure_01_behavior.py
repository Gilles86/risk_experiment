"""Figure 1 -- behavioural order x stake effects (panels B and C).

New Figure 1 of the NCOMMS-24-63995B revision is the behaviour figure. Two of
its four panels are data-driven and regenerated here:

  * **1B** -- proportion of risky choices vs safe-offer magnitude, hue = Order
    (Safe first / Risky first), SEM error bars across participants. Pure
    behaviour; no model. (Old notebook saved this as ``mag_order_effect.pdf``.)

  * **1C** -- psychophysical curves per stake size (Small / Medium / Large):
    proportion of risky choices vs (binned) risky/safe payoff ratio, hue =
    Order, with a risk-neutral-point (RNP) inset per stake. The curves are pure
    behaviour; the RNP insets are the group-level posteriors of the
    ``probit_full`` model (the published analysis). (Old notebook:
    ``stake_effect.pdf`` + ``rnp_stake_effect.pdf``.)

Panels 1A (trial-sequence schematic) and 1D (PMCM intuition cartoon) are
hand-drawn and are NOT regenerated here.

The expensive step -- building the bauer/bambi ``probit_full`` model and reading
its trace to derive the group-level RNP posteriors for the insets -- is cached
to a source-data TSV, so re-styling never re-runs it. Pass --recompute to force
it. Nothing here changes any number: the plotted quantities reproduce the
notebook exactly.

Usage
-----
    python -m risk_experiment.figures.figure_01_behavior
    python -m risk_experiment.figures.figure_01_behavior --recompute
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
from risk_experiment.utils.data import get_all_behavior


PANEL_B = 'figure_01_mag_order_effect'
PANEL_C = 'figure_01_rnp_stake_effect'

N_BINS = 7  # number of risky/safe ratio bins for panel 1C psychophysical curves
HUE_ORDER = ['Safe first', 'Risky first']
STAKE_ORDER = ['Small', 'Medium', 'Large']


# ---------------------------------------------------------------------------
# Behavioural data (cheap; recomputed every run)
# ---------------------------------------------------------------------------
def get_behavior(bids_folder='/data/ds-risk'):
    """Load all behaviour and add the binning columns used by panels B and C."""
    df = get_all_behavior(bids_folder=bids_folder)

    # Panel C: per-subject quantile bins of the risky/safe payoff ratio.
    df['bin(risky/safe)'] = (
        df.groupby('subject', group_keys=False)
        .apply(lambda x: pd.qcut(x['n_risky'] / x['n_safe'], q=N_BINS,
                                 labels=False, duplicates='drop'))
    )

    # Stake size = safe offer; tertiles -> Small / Medium / Large.
    df['bin(stake size)'] = pd.qcut(df['n_safe'], q=3, labels=STAKE_ORDER)

    return df


# ---------------------------------------------------------------------------
# Panel 1B -- prop. risky choices vs safe-offer magnitude
# ---------------------------------------------------------------------------
def source_data_B(df):
    """Per-(subject, Order, safe offer) proportion of risky choices."""
    tmp = (df.reset_index()
           .groupby(['subject', 'Order', 'n_safe'])['chose_risky']
           .mean()
           .reset_index())
    return tmp


def plot_B(df, bids_folder='/data/ds-risk'):
    import matplotlib.pyplot as plt

    style.set_style()
    tmp = source_data_B(df)

    # Panel 1B: exact Affinity box 42.4 x 37.9 mm (page size == figsize via
    # tight=False in save_panel; constrained_layout packs labels inside).
    fig, ax = plt.subplots(figsize=(42.4 * style.MM, 37.9 * style.MM),
                           constrained_layout=True)

    ax.axhline(0.5, c='k', ls='--', lw=0.6, zorder=0)

    sns.pointplot(data=tmp, x='n_safe', y='chose_risky', hue='Order',
                  hue_order=HUE_ORDER,
                  palette=style.ORDER_COLORS, dodge=.25, linestyle='none',
                  errorbar='se', err_kws={'linewidth': 1.0}, markersize=5,
                  ax=ax)

    ax.set_xlabel('Safe offer')
    # Short label so it fits the 38 mm panel height on one line.
    ax.set_ylabel('P(risky choice)')
    ax.set_yticks([0.4, 0.5, 0.6])   # include the 0.5 indifference line
    # Integer tick labels (the safe offers are 5, 7, 10, ... not 5.0, 7.0).
    ax.set_xticklabels([f'{int(float(t.get_text()))}'
                        for t in ax.get_xticklabels()])
    if ax.get_legend() is not None:
        ax.get_legend().remove()  # legend is a separate ingredient (placed in Affinity)

    sns.despine(ax=ax, offset=4, trim=False)

    pdf = style.save_panel(fig, PANEL_B, bids_folder, tight=False)
    print(f'Wrote figure: {pdf}')
    return fig


# ---------------------------------------------------------------------------
# Panel 1C -- psychophysical curves per stake + RNP insets
# ---------------------------------------------------------------------------
def source_data_C_curves(df):
    """Per-(subject, Order, stake, ratio-bin): proportion of risky choices and
    the mean log(risky/safe) of the bin (the x-position on the curve)."""
    tmp = (df.reset_index()
           .groupby(['subject', 'Order', 'bin(stake size)', 'bin(risky/safe)'],
                    observed=True)
           .agg(chose_risky=('chose_risky', 'mean'),
                logrs=('log(risky/safe)', 'mean'))
           .reset_index())
    return tmp


def _invprobit(x):
    return ss.norm.ppf(x)


def _get_fake_data(data, group):
    unique_subjects = data.index.unique(level='subject')
    subs = unique_subjects[:1] if group else unique_subjects
    fake = pd.MultiIndex.from_product(
        [subs, [0, 1], data['n_safe'].unique(), [False, True]],
        names=['subject', 'x', 'n_safe', 'risky_first']).to_frame(index=False)
    return fake


def _extract_intercept_gamma(trace, model, data, group=True):
    fake = _get_fake_data(data, group)
    pred = model.predict(trace, 'response_params', fake, inplace=False,
                         include_group_specific=not group)['posterior']['p']
    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake))

    pred0 = pred.xs(0, 0, 'x')
    intercept = pd.DataFrame(_invprobit(pred0), index=pred0.index,
                             columns=pred0.columns)
    gamma = _invprobit(pred.xs(1, 0, 'x')) - intercept

    intercept = intercept.droplevel(0, 1)
    gamma = gamma.droplevel(0, 1)
    return intercept, gamma


def compute_rnp(bids_folder='/data/ds-risk'):
    """Group-level RNP posterior per (Order, stake) from the probit_full model.

    Reproduces the notebook's ``get_info('probit_full', group=True)`` exactly:
    rnp = clip(exp(intercept / gamma), 0, 1) of the group-level probit fit.
    """
    from risk_experiment.cogmodels.fit_probit import build_model, get_data

    df = get_data('probit_full', session=None, bids_folder=bids_folder)
    model = build_model('probit_full', df=df, session=None,
                        bids_folder=bids_folder)
    idata = az.from_netcdf(op.join(bids_folder, 'derivatives', 'cogmodels',
                                   'model-probit_full_trace.netcdf'))
    df['x'] = df['log(risky/safe)']

    intercept, gamma = _extract_intercept_gamma(idata, model, df, group=True)
    rnp = np.clip(np.exp(intercept / gamma), 0, 1)
    rnp = rnp.stack([0, 1]).to_frame('rnp').reset_index()

    rnp['n_safe_bin'] = pd.qcut(rnp['n_safe'], q=3, labels=STAKE_ORDER)
    rnp['Order'] = rnp['risky_first'].map({True: 'Risky first',
                                           False: 'Safe first'})
    return rnp


def load_or_compute_rnp(bids_folder='/data/ds-risk', recompute=False):
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)
    tsv = op.join(sd_dir, f'{PANEL_C}_rnp.tsv')

    if recompute or not op.exists(tsv):
        rnp = compute_rnp(bids_folder)
        rnp.to_csv(tsv, sep='\t', index=False)
        print(f'Wrote source data: {tsv}')
    else:
        rnp = pd.read_csv(tsv, sep='\t')
        print(f'Loaded cached source data: {tsv}')
    return rnp


def _hdi(x, hdi_prob=0.95):
    return pd.Series(az.hdi(np.asarray(x), hdi_prob=hdi_prob),
                     index=['hdi_lower', 'hdi_upper'])


def plot_C(df, rnp, bids_folder='/data/ds-risk'):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style.set_style()
    curves = source_data_C_curves(df)

    # Two stacked rows per stake: the psychophysical curve on top, the RNP
    # (risk-neutral point) panel directly below it at the same width, so the RNP
    # reads against the p = 0.5 crossover of the curve above.
    fig, axes = plt.subplots(2, 3, figsize=(128.7 * style.MM, 54 * style.MM),
                             sharey='row',
                             gridspec_kw={'height_ratios': [2.6, 1.0]},
                             constrained_layout=True)
    # Shared x-axis for the two rows: log(risky/safe). The probit crosses 0.5
    # at log(risky/safe) = -log(RNP), so placing each order's RNP marker at
    # -log(RNP) lines it up exactly under that order's curve crossover. The
    # risk-neutral point (RNP = 0.55) sits at -log(0.55).
    NEUTRAL = -np.log(0.55)
    # Tightened to the data (curve log-ratios ~0.27-1.22, RNP HDIs ~0.39-1.30)
    # with small margins for the edge labels -- less dead space on the left.
    XLIM = (0.12, 1.42)
    # Top curve x-axis is log(risky/safe); label it with natural-space ratios at
    # log(ratio). Nice ratios inside exp(XLIM) ~ (1.13, 4.14).
    ratio_ticks = [1.5, 2, 3, 4]
    rnp_ticks = [0.7, 0.55, 0.4]
    # Safe-offer range of each stake tertile (e.g. Small = 5-7), data-driven.
    stake_rng = (df.reset_index()
                 .groupby('bin(stake size)', observed=True)['n_safe']
                 .agg(['min', 'max']))

    for j, stake in enumerate(STAKE_ORDER):
        ax, rax = axes[0, j], axes[1, j]
        sub = curves[curves['bin(stake size)'] == stake]

        # --- top: psychophysical curve vs log(risky/safe), points joined ---
        ax.axhline(0.5, ls='--', c='k', lw=0.6, zorder=0)
        ax.axvline(NEUTRAL, ls='--', c='k', lw=0.6, zorder=0)
        for order in HUE_ORDER:
            agg = (sub[sub['Order'] == order]
                   .groupby('bin(risky/safe)')
                   .agg(mean=('chose_risky', 'mean'),
                        sem=('chose_risky', 'sem'),
                        x=('logrs', 'mean'))
                   .reset_index().sort_values('x'))
            color = style.ORDER_COLORS[order]
            ax.plot(agg['x'], agg['mean'], '-', color=color, lw=1.1, zorder=2)
            ax.errorbar(agg['x'], agg['mean'], yerr=agg['sem'], fmt='o', ms=5,
                        color=color, mec='none', elinewidth=1.0, capsize=0,
                        zorder=3)
        _lo, _hi = int(stake_rng.loc[stake, 'min']), int(stake_rng.loc[stake, 'max'])
        ax.set_title(f'{stake} stakes ({_lo}–{_hi})', fontweight='bold', pad=6)
        ax.set_xlim(*XLIM)
        # Natural-space ratio ticks at log(ratio) (curve x-axis is in log units).
        ax.set_xticks([np.log(t) for t in ratio_ticks])
        ax.set_xticklabels([f'{t:g}' for t in ratio_ticks])
        # Single small label centred under the middle panel == centred under all
        # three (three equal columns); constrained_layout reserves its space.
        ax.set_xlabel('Risky / safe payoff', fontsize=7)
        ax.set_yticks([0.25, 0.5, 0.75])     # data sits ~0.1-0.85; no need for 0
        ax.set_ylim(0.06, 0.92)
        ax.set_ylabel('P(risky choice)' if j == 0 else '')
        sns.despine(ax=ax, offset=3, trim=False)

        # --- bottom: RNP per order at -log(RNP) (same x-axis as the curve) ---
        rsub = rnp[rnp['n_safe_bin'] == stake]
        rax.axvline(NEUTRAL, ls='--', c='k', lw=0.6, zorder=0)
        for k, order in enumerate(HUE_ORDER):
            vals = -np.log(rsub.loc[rsub['Order'] == order, 'rnp'].values)
            m = vals.mean()
            lo, hi = az.hdi(vals)
            rax.errorbar(m, -k, xerr=[[m - lo], [hi - m]], fmt='o', ms=4,
                         color=style.ORDER_COLORS[order], elinewidth=1.4,
                         capsize=0)
        rax.set_xlim(*XLIM)
        rax.set_xticks([-np.log(t) for t in rnp_ticks])   # position = -log(RNP)
        rax.set_xticklabels([f'{t:g}'.lstrip('0') for t in rnp_ticks])  # = RNP
        rax.set_ylim(-1.9, 0.9)
        rax.set_yticks([])
        # One small centred 'RNP' label under all three (middle panel only).
        rax.set_xlabel('RNP', fontsize=7)
        # RNP > .55 (left of neutral) = risk-seeking; < .55 (right) = risk-averse.
        rax.text(XLIM[0] + 0.05, 0.8, 'Risk-seeking', fontsize=6, ha='left',
                 va='bottom', color='0.45')
        rax.text(XLIM[1] - 0.05, 0.8, 'Risk-averse', fontsize=6, ha='right',
                 va='bottom', color='0.45')
        sns.despine(ax=rax, left=True, offset=3, trim=False)
        rax.tick_params(axis='y', length=0)

    # No in-panel key -- the order legend is a separate ingredient (below),
    # placed by hand in Affinity so it never overlaps the small panels.

    pdf = style.save_panel(fig, PANEL_C, bids_folder, tight=False)
    print(f'Wrote figure: {pdf}')
    return fig


def make_order_legend(bids_folder='/data/ds-risk'):
    """Standalone 'Safe first / Risky first' key as its own little PDF, to be
    placed by hand in the assembled figure (avoids overlap on small panels)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style.set_style()
    fig, ax = plt.subplots(figsize=(34 * style.MM, 12 * style.MM))
    ax.axis('off')
    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=5,
                      color=style.ORDER_COLORS[k]) for k in HUE_ORDER]
    # Boxed so the key reads as its own element in the assembled figure.
    ax.legend(handles, HUE_ORDER, loc='center', frameon=True, fontsize=7,
              handletextpad=0.3, labelspacing=0.4, borderpad=0.6,
              edgecolor='0.6', facecolor='white', framealpha=1.0)
    # Tight crop + transparent page background (only the boxed key remains),
    # so it drops cleanly onto the Affinity canvas.
    pdf = style.save_panel(fig, 'figure_01_order_legend', bids_folder,
                           tight=True, transparent=True, pad=0.04)
    print(f'Wrote figure: {pdf}')
    return fig


# ---------------------------------------------------------------------------
def write_source_data(df, bids_folder='/data/ds-risk'):
    """Export tidy source-data TSVs for the two behavioural panels."""
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)

    b = source_data_B(df)
    b.to_csv(op.join(sd_dir, f'{PANEL_B}.tsv'), sep='\t', index=False)
    print(f'Wrote source data: {op.join(sd_dir, f"{PANEL_B}.tsv")}')

    c = source_data_C_curves(df)
    c.to_csv(op.join(sd_dir, f'{PANEL_C}.tsv'), sep='\t', index=False)
    print(f'Wrote source data: {op.join(sd_dir, f"{PANEL_C}.tsv")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-folder', default='/data/ds-risk')
    parser.add_argument('--recompute', action='store_true',
                        help='Force recomputation of the RNP posteriors.')
    args = parser.parse_args()

    df = get_behavior(args.bids_folder)
    rnp = load_or_compute_rnp(args.bids_folder, recompute=args.recompute)

    write_source_data(df, args.bids_folder)
    plot_B(df, args.bids_folder)
    plot_C(df, rnp, args.bids_folder)
    make_order_legend(args.bids_folder)
