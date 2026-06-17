"""Figure 4 -- ELPD model comparison (the split-out former panel 3E).

The editor asked us to split the model-comparison forest plot out of old
Figure 3 so it can breathe as a standalone figure. This script

  1. computes the PSIS-LOO comparison over the five behavioural models
     (identical to the old `model_comparison.ipynb`), caching the
     `az.compare` table to a source-data TSV, and
  2. renders the standalone forest panel in the house style.

The expensive step (building the bauer models + `compute_log_likelihood`
+ LOO) is cached, so re-styling never re-runs it. Pass --recompute to
force it.

Content is unchanged from the published analysis: the plotted quantities
(ELPD +/- SE filled circles; ELPD-difference +/- dSE grey triangles for
non-best models; dashed line at the best model) reproduce
`arviz.plot_compare` exactly.

Usage
-----
    python -m risk_experiment.figures.figure_04_model_comparison
    python -m risk_experiment.figures.figure_04_model_comparison --recompute
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns

from risk_experiment.figures import style


# (model_label, display name). Order is irrelevant -- az.compare sorts by ELPD.
MODELS = [
    ('klw', 'Model A: Shared prior, equal noise'),
    ('12', 'PMCM model'),
    ('42', 'Model C: Shared prior, varying noise'),
    ('52', 'Model B: Varying priors, equal noise'),
    ('eu', 'Model D: Expected utility model'),
]

PANEL_NAME = 'figure_04_model_comparison'


def compute_comparison(bids_folder='/data/ds-risk'):
    """Build models, compute log-likelihood, return the az.compare table."""
    import pymc as pm
    style.shim_pymc_for_bauer(pm)
    from risk_experiment.cogmodels.fit_model import build_model, get_data

    idatas = {}
    for model_label, model_name in MODELS:
        df = get_data(model_label, None, bids_folder, None)
        model = build_model(model_label, df, None)
        model.build_estimation_model()

        idata = az.from_netcdf(
            op.join(bids_folder, 'derivatives', 'cogmodels',
                    f'model-{model_label}_trace.netcdf'))

        if 'log_likelihood' not in idata:
            with model.estimation_model:
                pm.compute_log_likelihood(idata)

        idatas[model_name] = idata

    return az.compare(idatas)


def load_or_compute(bids_folder='/data/ds-risk', recompute=False):
    """Return the comparison table, caching it to the source-data TSV."""
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)
    tsv = op.join(sd_dir, f'{PANEL_NAME}.tsv')

    if recompute or not op.exists(tsv):
        comp = compute_comparison(bids_folder)
        comp.to_csv(tsv, sep='\t')
        print(f'Wrote source data: {tsv}')
    else:
        comp = pd.read_csv(tsv, sep='\t', index_col=0)
        print(f'Loaded cached source data: {tsv}')

    return comp


def plot(comp, bids_folder='/data/ds-risk'):
    """Standalone ELPD forest plot, reproducing az.plot_compare quantities."""
    import matplotlib.pyplot as plt

    style.set_style()

    # Sort best (highest ELPD) on top, exactly like az.plot_compare.
    comp = comp.sort_values('elpd_loo', ascending=False)
    n = comp.shape[0]

    # az.plot_compare's y layout: model rows at even positions, the
    # ELPD-difference markers nudged half a step between rows.
    yticks_pos, step = np.linspace(0, -1, (n * 2) - 1, retstep=True)
    yticks_pos[1::2] = yticks_pos[1::2] + step / 2

    # Single column (88 mm) -- a standard Nature Communications width.
    fig, ax = plt.subplots(figsize=(style.WIDTH_SINGLE, 2.3),
                           constrained_layout=True)

    lw = 0.8

    # ELPD +/- SE -- white-faced circles at each model.
    ax.errorbar(x=comp['elpd_loo'], y=yticks_pos[::2], xerr=comp['se'],
                fmt='o', color='k', mfc='white', mew=lw, elinewidth=lw,
                ms=4.5, label='ELPD', zorder=3)

    # ELPD difference +/- dSE -- grey triangles for the non-best models,
    # positioned between rows (same x as the ELPD marker, smaller error bar).
    ax.errorbar(x=comp['elpd_loo'].iloc[1:], y=yticks_pos[1::2],
                xerr=comp['dse'].iloc[1:], fmt='^', color='0.45', mew=lw,
                elinewidth=lw, ms=4, label='ELPD difference', zorder=2)

    # Dashed reference line at the best model's ELPD.
    ax.axvline(comp['elpd_loo'].iloc[0], ls='--', color='0.6', lw=lw, zorder=0)

    ax.set_yticks(yticks_pos[::2])
    # Short labels fit the single column; full model descriptions live in the
    # caption (and Figure 3's column headers).
    ax.set_yticklabels([m.split(':')[0] for m in comp.index])
    ax.set_ylim(-1 + step, 0 - step)
    ax.set_xlabel('ELPD (higher is better)')

    # In-panel key (replaces the caption-level legend), in a light box to match
    # the standalone legend ingredients (order / risk-category keys).
    ax.legend(loc='upper left', frameon=True, fontsize=8, handletextpad=0.4,
              edgecolor='0.6', facecolor='white', framealpha=1.0,
              borderpad=0.6).get_frame().set_linewidth(0.6)

    sns.despine(ax=ax, offset=4, trim=False, left=True)
    ax.tick_params(axis='y', length=0)

    pdf = style.save_panel(fig, PANEL_NAME, bids_folder)
    print(f'Wrote figure: {pdf}')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-folder', default='/data/ds-risk')
    parser.add_argument('--recompute', action='store_true',
                        help='Force recomputation of the LOO comparison table.')
    args = parser.parse_args()

    comp = load_or_compute(args.bids_folder, recompute=args.recompute)
    plot(comp, args.bids_folder)
