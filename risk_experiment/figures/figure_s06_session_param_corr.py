"""Supplementary Figure 6 -- cross-session (3T vs 7T) correlation of the
separately-estimated PMCM parameter estimates.

Each participant's six group-model (per-session ``model-1``) parameters are
summarised by their posterior mean, separately for the 3T and the 7T session.
This panel is the 6x6 matrix of Pearson correlations between the 3T estimates
(rows) and the 7T estimates (columns), across participants.

The scientifically meaningful quantity is the **diagonal**: same parameter, 3T
vs 7T -- i.e. the test-retest reproducibility of each parameter across scanners.

    NOTE ON THE PUBLISHED VERSION. The original supplementary figure was drawn
    from the *full* ``parameter_estimates.corr()`` (a 12x12 matrix over all
    parameter x session columns), whose diagonal is every column correlated with
    *itself* -- trivially 1.0 and uninformative. This script instead plots the
    3T(rows) x 7T(cols) off-diagonal block (the ``corr_sessions`` selection in
    ``cogmodels/notebooks/parameter_recovery.ipynb``), so the diagonal shows the
    genuine cross-session correlations (all moderate-to-strong positive, none 1).

Reproduces the notebook computation exactly, except the parameter names are the
current trace names (``risky_prior_sd`` / ``safe_prior_sd``; the notebook's
``*_std`` no longer exist in the traces).

The 6x6 matrix is cached to a source-data TSV; restyling never re-reads the
traces. Pass ``--recompute`` to force it.

Usage:
    python -m risk_experiment.figures.figure_s06_session_param_corr
    python -m risk_experiment.figures.figure_s06_session_param_corr --recompute
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd

from risk_experiment.figures import style

BIDS = '/data/ds-risk'
SESSIONS = [('3t2', '3T'), ('7t2', '7T')]

# Read the SAME per-session traces the authoritative statistics use
# (revision/report_statistics.py -> Supplementary Table 1), so this figure's
# correlations match the reported test-retest values to full precision. The
# original manuscript-era ses-*_model-1 traces were lost; revision_refit is the
# clean regeneration (see revision/refit_per_session.py) and the agreed source
# of truth for the per-session PMCM numbers.
TRACE_SUBDIR = op.join('derivatives', 'cogmodels', 'revision_refit')

# Subject-level posterior variables (the current trace names).
PARAMS = ['n1_evidence_sd', 'n2_evidence_sd',
          'risky_prior_mu', 'safe_prior_mu',
          'risky_prior_sd', 'safe_prior_sd']

# Readable single-line tick labels (first letter capitalised; Unicode Greek
# + subscripts) -- narrow enough that rotated x-labels don't collide.
PARAM_LABELS = {
    'n1_evidence_sd': 'Evidence SD₁',
    'n2_evidence_sd': 'Evidence SD₂',
    'risky_prior_mu': 'Risky prior μ',
    'safe_prior_mu':  'Safe prior μ',
    'risky_prior_sd': 'Risky prior σ',
    'safe_prior_sd':  'Safe prior σ',
}


# --------------------------------------------------------------------------- #
# Computation (cheap: subject-level posterior means from the two netcdfs)
# --------------------------------------------------------------------------- #
def _session_means(session, bids_folder):
    """Per-subject posterior-mean of each parameter for one session, from the
    revision_refit traces (the source of truth used by report_statistics.py)."""
    idata = az.from_netcdf(op.join(bids_folder, TRACE_SUBDIR,
                                   f'ses-{session}_model-1_trace.netcdf'))
    return idata.posterior[PARAMS].to_dataframe().groupby('subject').mean()[PARAMS]


def compute_corr(bids_folder):
    """The 3T(rows) x 7T(cols) cross-session correlation matrix + N.

    Reproduces ``parameter_estimates.corr().loc[(:, '3t'), (:, '7t')]`` from the
    notebook, restricted to participants with complete data in both sessions so a
    single N applies to every cell.
    """
    m3 = _session_means('3t2', bids_folder)
    m7 = _session_means('7t2', bids_folder)
    common = m3.index.intersection(m7.index)
    m3, m7 = m3.loc[common], m7.loc[common]

    # corr(param_i @ 3T, param_j @ 7T) for every (i, j).
    corr = pd.DataFrame(index=PARAMS, columns=PARAMS, dtype=float)
    for pi in PARAMS:
        for pj in PARAMS:
            corr.loc[pi, pj] = np.corrcoef(m3[pi].values, m7[pj].values)[0, 1]
    corr.index.name = 'param_3T'
    corr.columns.name = 'param_7T'
    return corr, len(common)


def load_corr(bids_folder, recompute=False):
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)
    tsv = op.join(sd_dir, 'figure_s06_session_param_corr.tsv')
    if recompute or not op.exists(tsv):
        corr, n = compute_corr(bids_folder)
        # Stash N in the header comment line so the source-data file is self-
        # documenting without adding a column to a square matrix.
        with open(tsv, 'w') as f:
            f.write(f'# Pearson r between per-participant posterior-mean PMCM '
                    f'parameters, 3T (rows) vs 7T (cols); n = {n} participants\n')
            corr.to_csv(f, sep='\t')
        print(f'Wrote source data: {tsv}  (n = {n})')
    else:
        with open(tsv) as f:
            header = f.readline()
        n = int(header.split('n =')[1].split('participants')[0])
        corr = pd.read_csv(tsv, sep='\t', comment='#', index_col=0)
        corr = corr.loc[PARAMS, PARAMS]
    return corr, n


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(corr, n):
    """3T x 7T cross-session reliability, as a diverging (blue/red) correlation
    matrix centred at 0 -- these correlations are signed (off-diagonals go
    negative), so a diverging map reads correctly. Plots the correct 3T x 7T
    block, so the diagonal shows the real cross-session correlations instead of
    the trivial 1.0s of the full ``.corr()``."""
    import matplotlib.pyplot as plt
    style.set_style()

    M = corr.loc[PARAMS, PARAMS].values.astype(float)
    k = len(PARAMS)
    VMIN, VMAX = -1.0, 1.0

    fig, ax = plt.subplots(figsize=(style.WIDTH_ONEHALF + 0.6, 4.5))
    fig.subplots_adjust(left=0.22, right=0.87, top=0.9, bottom=0.28)

    im = ax.imshow(M, cmap='RdBu_r', vmin=VMIN, vmax=VMAX, aspect='equal')

    # Per-cell r annotations; white on the dark (saturated) cells, near-black on
    # the pale cells around zero.
    for i in range(k):
        for j in range(k):
            r = M[i, j]
            ax.text(j, i, f'{r:.2f}', ha='center', va='center', fontsize=7.5,
                    color='white' if abs(r) > 0.5 else '0.12')

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([PARAM_LABELS[p] for p in PARAMS], rotation=35,
                       ha='right', rotation_mode='anchor', fontsize=8)
    ax.set_yticklabels([PARAM_LABELS[p] for p in PARAMS], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel('7T session', labelpad=6)
    ax.set_ylabel('3T session', labelpad=6)

    ax.text(1.0, -0.28, f'n = {n} participants', transform=ax.transAxes,
            ha='right', va='top', fontsize=7.5, color='0.35')

    # Slim colourbar, symmetric diverging scale centred at 0.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label('Pearson r', labelpad=4)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(width=0.6, length=3)

    return fig


def main(bids_folder=BIDS, recompute=False):
    corr, n = load_corr(bids_folder, recompute=recompute)
    fig = build_figure(corr, n)

    out_dir = style.figures_dir()
    os.makedirs(out_dir, exist_ok=True)
    pdf = op.join(out_dir, 'figure_s06_session_param_corr.pdf')
    png = op.join(out_dir, 'figure_s06_session_param_corr.png')
    style.save_panel(fig, 'figure_s06_session_param_corr')
    fig.savefig(png, dpi=300)
    print(f'Wrote figure: {pdf}')
    print(f'Wrote figure: {png}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-folder', default=BIDS)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    main(args.bids_folder, recompute=args.recompute)
