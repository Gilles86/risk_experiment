"""Figure 3 -- posterior-predictive checks of the four ALTERNATIVE models.

Old Figure 3 was split: panels A-D (the posterior-predictive grids for the
four alternative behavioural models) stay as the new Figure 3, while old panel
E (the ELPD forest plot) became the standalone Figure 4
(``figure_04_model_comparison.py``). This script owns A-D only.

Each panel is a 3-stake-size x order posterior-predictive grid:
markers = data (proportion chose risky, binned by risky/safe payoff ratio),
line + shaded band = model posterior prediction with its 95% HDI. One clean
PDF (+ SVG) per alternative model:

  Model A: Shared prior, equal noise   -> klw  (gray: model collapses orders)
  Model B: Varying priors, equal noise -> 52   (gray)
  Model C: Shared prior, varying noise -> 42   (coloured by order)
  Model D: Expected utility model      -> eu   (gray)

The PMCM ppc itself is Figure 2A and is NOT produced here.

The expensive step (build the bauer model + ``model.ppc`` posterior-predictive
sampling) is cached per model to a source-data TSV: a tidy table of the plotted
quantities (data mean + 95% HDI of the predicted proportion, per stake size x
order x ratio bin). Re-styling never re-runs the sampling. Pass --recompute to
force it.

Content is unchanged from the published analysis: the aggregation reproduces
``risk_experiment/notebooks/revision1/ppcs.ipynb`` (``get_ppc`` + ``plot_ppc``)
exactly -- same binning (3 stake-size qcut, 7 risky/safe qcut per subject),
same thinning (every 20th draw), same 95% az.hdi.

Usage
-----
    python -m risk_experiment.figures.figure_03_alt_models
    python -m risk_experiment.figures.figure_03_alt_models --recompute
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns

from risk_experiment.figures import style


# (model_label, display title, gray). ``gray`` matches ppcs.ipynb: models that
# cannot distinguish the two presentation orders are drawn in a single gray,
# the one model that can (Model C) keeps the Safe/Risky-first order colours.
MODELS = [
    ('klw', 'Model A: Shared prior, equal noise', True),
    ('52', 'Model B: Varying priors, equal noise', True),
    ('42', 'Model C: Shared prior, varying noise', False),
    ('eu', 'Model D: Expected utility model', True),
]

STAKE_ORDER = ['Small', 'Medium', 'Large']
STAKE_N = {'Small': '5, 7', 'Medium': '10, 14', 'Large': '20, 28'}
# Semantic order colours (carry meaning the reader has learned) -- do not change.
ORDER_LABELS = {False: 'Safe first', True: 'Risky first'}
ORDER_COLORS = style.ORDER_COLORS  # {'Safe first': '#1f77b4', 'Risky first': '#ff7f0e'}


def _panel_name(model_label):
    letter = {'klw': 'modelA_klw', '52': 'modelB_52',
              '42': 'modelC_42', 'eu': 'modelD_eu'}[model_label]
    return f'figure_03_{letter}'


def _get_hdi(x, hdi_prob=0.95):
    return pd.Series(az.hdi(x.values, hdi_prob=hdi_prob),
                     index=['hdi_lower', 'hdi_upper']).T


def compute_summary(model_label, bids_folder='/data/ds-risk'):
    """Build the bauer model, draw the posterior-predictive, and reduce it to a
    tidy summary table reproducing ppcs.ipynb's ``get_ppc`` + ``plot_ppc``.

    Returns a long DataFrame with one row per
    (stake size x order x ratio bin): the data mean (``chose_risky``) and the
    posterior-predictive mean + 95% HDI band (``pred_mean``/``hdi_lower``/
    ``hdi_upper``).
    """
    import pymc as pm
    style.shim_pymc_for_bauer(pm)
    from risk_experiment.cogmodels.fit_model import build_model, get_data

    df = get_data(model_label=model_label, session=None,
                  bids_folder=bids_folder, roi=None)
    model = build_model(model_label=model_label, df=df, roi=None)

    idata = az.from_netcdf(
        op.join(bids_folder, 'derivatives', 'cogmodels',
                f'model-{model_label}_trace.netcdf'))

    # --- get_ppc (verbatim logic from the notebook) ---------------------
    ppc = model.ppc(idata=idata.sel(draw=slice(None, None, 20)),
                    paradigm=df.drop('session', axis=1),
                    var_names=['ll_bernoulli'])
    ppc = ppc.stack([0, 1]).to_frame('choice_pred')
    ppc = ppc.reset_index().set_index(
        df.index.names + ['chain', 'draw'])[['choice_pred']]
    ppc = ppc.join(df[['risky_first', 'n_risky', 'n_safe', 'chose_risky']])

    # When the risky option was presented first, flip the predicted choice to
    # "chose risky". The notebook wrote ``~choice_pred`` assuming a boolean; the
    # ppc here is numeric (0/1), for which ``1 - choice_pred`` is the identical
    # logical-not (``~0`` would wrongly give -1).
    ppc['chose_risky_pred'] = np.where(
        ppc['risky_first'], 1 - ppc['choice_pred'], ppc['choice_pred'])

    n_bins = 7
    ppc['bin(n_safe)'] = pd.qcut(ppc['n_safe'], q=3,
                                 labels=['Small', 'Medium', 'Large'])
    ppc['bin(risky/safe)'] = ppc.groupby('subject').apply(
        lambda x: pd.qcut(x['n_risky'] / x['n_safe'], q=n_bins,
                          labels=False, duplicates='drop')
    ).reset_index(level=0, drop=True)

    # --- plot_ppc aggregation (verbatim) --------------------------------
    tmp = ppc.groupby(['bin(n_safe)', 'risky_first', 'bin(risky/safe)',
                       'chain', 'draw'])[['chose_risky_pred', 'chose_risky']].mean()

    # 95% HDI of the predicted proportion across (chain, draw), per cell.
    grp = tmp.reset_index().groupby(['bin(n_safe)', 'risky_first', 'bin(risky/safe)'])
    hdi = grp['chose_risky_pred'].apply(lambda x: _get_hdi(x)).unstack()
    pred_mean = grp['chose_risky_pred'].mean().rename('pred_mean')
    data_mean = grp['chose_risky'].mean().rename('chose_risky')

    summary = pd.concat([data_mean, pred_mean, hdi], axis=1).reset_index()
    summary['model_label'] = model_label
    return summary


def load_or_compute(model_label, bids_folder='/data/ds-risk', recompute=False):
    """Return the per-model summary table, caching it to the source-data TSV."""
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)
    tsv = op.join(sd_dir, f'{_panel_name(model_label)}.tsv')

    if recompute or not op.exists(tsv):
        summary = compute_summary(model_label, bids_folder)
        summary.to_csv(tsv, sep='\t', index=False)
        print(f'Wrote source data: {tsv}')
    else:
        summary = pd.read_csv(tsv, sep='\t')
        print(f'Loaded cached source data: {tsv}')

    return summary


def plot(model_label, title, gray, summary, bids_folder='/data/ds-risk'):
    """Render one alternative-model posterior-predictive grid in house style."""
    import matplotlib.pyplot as plt

    summary = summary.copy()
    summary['bin(n_safe)'] = pd.Categorical(summary['bin(n_safe)'],
                                            categories=STAKE_ORDER, ordered=True)

    # 3 stake-size columns side by side; double-column-ish row of small panels.
    # ~50 mm tall so the four model panels stack within A4 height (4x50 + gaps
    # <= 230 mm) when assembled as the full Figure 3 column.
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_DOUBLE, 2.0),
                             sharey=True, constrained_layout=True)

    handles = {}
    for ax, stake in zip(axes, STAKE_ORDER):
        sub = summary[summary['bin(n_safe)'] == stake]
        ax.axhline(0.5, c='k', ls='--', lw=0.5, zorder=0)
        ax.axvline(3, c='k', ls='--', lw=0.5, zorder=0)

        for risky_first in [False, True]:
            cell = sub[sub['risky_first'] == risky_first].sort_values('bin(risky/safe)')
            if cell.empty:
                continue
            label = ORDER_LABELS[risky_first]
            # Data always carries the order colour (blue/orange); for models
            # that collapse the two orders, the *prediction* is drawn gray so
            # the eye sees the data split the model cannot reproduce.
            data_color = ORDER_COLORS[label]
            model_color = '0.45' if gray else ORDER_COLORS[label]
            x = cell['bin(risky/safe)'].values

            ax.fill_between(x, cell['hdi_lower'], cell['hdi_upper'],
                            color=model_color, alpha=0.15, lw=0, zorder=1)
            ax.plot(x, cell['pred_mean'], color=model_color, lw=1.1, zorder=2)
            h = ax.plot(x, cell['chose_risky'], marker='o', ls='', ms=3.5,
                        color=data_color, mec='none', zorder=3)[0]
            handles[label] = h

        ax.set_title(f'Stake size {stake}')
        # Plain text (no mathtext): keeps the panel strictly Helvetica -- a
        # $\frac{}{}$ label would pull in STIX glyphs for the fraction.
        ax.set_xlabel('Risky / safe payoff')
        ax.set_xticks([])
        ax.set_yticks([0, .25, .5, .75, 1.])
        ax.set_ylim(-0.03, 1.03)

    axes[0].set_ylabel('P(risky choice)')

    # In-panel key (not a caption legend). Data is colour-coded by order in
    # every panel now, so always show the order key. For order-collapsing
    # models, also note that the gray prediction does not separate the orders.
    order_h = [handles[lbl] for lbl in ['Safe first', 'Risky first']
               if lbl in handles]
    order_l = [lbl for lbl in ['Safe first', 'Risky first'] if lbl in handles]
    axes[0].legend(order_h, order_l, loc='upper left', frameon=False,
                   fontsize=8, handletextpad=0.3, borderaxespad=0.2)
    if gray:
        axes[2].annotate('Model: both orders', xy=(0.97, 0.04),
                         xycoords='axes fraction', ha='right', va='bottom',
                         fontsize=8, color='0.45')

    fig.suptitle(title, fontsize=8, fontweight='bold')

    sns.despine(fig=fig, offset=3, trim=False)
    for ax in axes:
        ax.tick_params(axis='x', length=0)

    name = _panel_name(model_label)
    pdf = style.save_panel(fig, name, bids_folder)
    print(f'Wrote figure: {pdf}')
    return fig


# Bold row labels for the composite (one per model, shown left of each row).
ROW_TITLE = {   # single line -> used as a left-aligned header above each row
    'klw': 'Model A: shared prior, equal noise',
    '52': 'Model B: varying priors, equal noise',
    '42': 'Model C: shared prior, varying noise',
    'eu': 'Model D: expected utility',
}


def plot_composite(summaries, bids_folder='/data/ds-risk'):
    """Single composite: stake sizes on columns, models on rows (4 x 3 grid).

    Built from one matplotlib *subfigure per model row*: each row carries its
    own bold model-name header (a subfigure suptitle), so the header always sits
    in its own band and never collides with the stake column titles. Panels are
    square, the whole thing is one journal column (88 mm) wide. ``summaries`` is
    a dict model_label -> summary table.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style.set_style()
    nrow, ncol = len(MODELS), len(STAKE_ORDER)
    # 3 ticks (not 4): the narrow one-column panels crowd 2.5 into 3.
    xpos, xlab = style.ratio_bin_ticks(bids_folder, nice=(1.5, 2, 3))
    x_neutral = style.ratio_pos(1 / 0.55, bids_folder)  # risk-neutral ratio

    # One journal column wide; tall enough for 4 square panel rows + a header
    # band per row + a bottom legend strip.
    fig = plt.figure(figsize=(style.WIDTH_SINGLE, 5.7), constrained_layout=True)
    # Rows 0 and 3 carry extra furniture (stake titles / xlabel); give them a
    # touch more height so every square panel ends up the same size.
    subfigs = fig.subfigures(nrow + 1, 1,
                             height_ratios=[1.12, 1.0, 1.0, 1.08, 0.30])

    for i, (model_label, _title, gray) in enumerate(MODELS):
        sf = subfigs[i]
        # Bold model-name header for this row, centred in its own band above
        # the panels.
        sf.suptitle(ROW_TITLE[model_label], ha='center',
                    fontweight='bold', fontsize=8)
        axes = sf.subplots(1, ncol, sharey=True)
        summ = summaries[model_label].copy()
        summ['bin(n_safe)'] = pd.Categorical(summ['bin(n_safe)'],
                                             categories=STAKE_ORDER, ordered=True)
        for j, (ax, stake) in enumerate(zip(axes, STAKE_ORDER)):
            ax.set_box_aspect(1)            # square panels
            sub = summ[summ['bin(n_safe)'] == stake]
            ax.axhline(0.5, c='k', ls='--', lw=0.5, zorder=0)
            ax.axvline(x_neutral, c='k', ls='--', lw=0.5, zorder=0)
            for risky_first in [False, True]:
                cell = sub[sub['risky_first'] == risky_first].sort_values(
                    'bin(risky/safe)')
                if cell.empty:
                    continue
                label = ORDER_LABELS[risky_first]
                data_color = ORDER_COLORS[label]
                model_color = '0.45' if gray else ORDER_COLORS[label]
                x = cell['bin(risky/safe)'].values
                ax.fill_between(x, cell['hdi_lower'], cell['hdi_upper'],
                                color=model_color, alpha=0.15, lw=0, zorder=1)
                ax.plot(x, cell['pred_mean'], color=model_color, lw=1.0, zorder=2)
                ax.plot(x, cell['chose_risky'], marker='o', ls='', ms=3.0,
                        color=data_color, mec='none', zorder=3)
            ax.set_xticks(xpos)
            ax.set_xticklabels(xlab)          # ratio ticks on every row: equal
            ax.set_yticks([0, .5, 1.])         # furniture -> equal-size squares
            ax.set_ylim(-0.04, 1.04)
            if i == 0:                       # stake = column title (top row only)
                ax.set_title(f'{stake} stakes\n(n = {STAKE_N[stake]})',
                             fontsize=7.5)
            if j == 0:                       # repeat y-label on the left column
                ax.set_ylabel('P(risky choice)', fontsize=6.5)
            if i == nrow - 1 and j == 1:     # one small xlabel, bottom middle
                ax.set_xlabel('Risky / safe', fontsize=6.5)
        sns.despine(ax=axes[0], offset=2, trim=False)
        for ax in axes:
            sns.despine(ax=ax, offset=2, trim=False)

    # Bottom legend strip (its own subfigure band).
    handles = [Line2D([0], [0], marker='o', ls='', ms=5,
                      color=ORDER_COLORS[lbl]) for lbl in ['Safe first', 'Risky first']]
    # Boxed, stacked key (matching the other figures' order legends).
    leg = subfigs[-1].legend(handles, ['Safe first', 'Risky first'], loc='center',
                             ncol=1, frameon=True, fontsize=8, edgecolor='0.6',
                             facecolor='white', framealpha=1.0, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.6)

    pdf = style.save_panel(fig, 'figure_03_composite', bids_folder)
    print(f'Wrote figure: {pdf}')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-folder', default='/data/ds-risk')
    parser.add_argument('--recompute', action='store_true',
                        help='Force recomputation of the posterior-predictive.')
    parser.add_argument('--model', default=None,
                        help='Only (re)build this model_label (klw/52/42/eu).')
    parser.add_argument('--individual', action='store_true',
                        help='Also emit the four separate per-model panels.')
    args = parser.parse_args()

    style.set_style()

    summaries = {}
    for model_label, title, gray in MODELS:
        if args.model is not None and model_label != args.model:
            continue
        summaries[model_label] = load_or_compute(model_label, args.bids_folder,
                                                 recompute=args.recompute)
        if args.individual:
            plot(model_label, title, gray, summaries[model_label],
                 args.bids_folder)

    if len(summaries) == len(MODELS):
        plot_composite(summaries, args.bids_folder)
