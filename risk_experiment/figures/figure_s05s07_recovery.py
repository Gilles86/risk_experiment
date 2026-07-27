"""Supplementary parameter-recovery correlation matrices (generating vs
estimated), restyled to match the other SI correlation matrices.

Two recovery "strategies" are simulated (``cogmodels/notebooks/
parameter_recovery.ipynb``): for each, 10 datasets are simulated from known
("generating") parameters, the PMCM is refit, and the per-subject posterior-mean
estimate is correlated with the generating value. The matrix is the mean (over
the 10 simulations) Pearson correlation between every generating parameter (rows)
and every estimated parameter (columns); the **diagonal is recoverability**.

  * ``figure_s05_recovery``  -- generating parameters sampled from the *data*
    (the fitted per-subject estimates), with the calibrated experimental design.
    Diagonal recoverability 0.73-0.93 (manuscript "0.74 to 0.93").
  * ``figure_s07_recovery``  -- generating parameters sampled from the *prior*.

Signed off-diagonals (parameter trade-offs) go negative, so a diverging blue/red
map centred at 0 is used -- matching Supplementary Fig. 6.

Reproduces the notebook computation exactly. Each matrix is cached to a
source-data TSV; restyling never re-reads the simulation posteriors. Pass
``--recompute`` to force it.

Usage:
    python -m risk_experiment.figures.figure_s05s07_recovery
    python -m risk_experiment.figures.figure_s05s07_recovery --recompute
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd

from risk_experiment.figures import style

BIDS = '/data/ds-risk'
RECOVERY_DIR = op.join('derivatives', 'parameter_recovery')

# Recovery sims use the analysis-era ``*_std`` parameter names.
PARAMS = ['n1_evidence_sd', 'n2_evidence_sd', 'risky_prior_mu', 'safe_prior_mu',
          'risky_prior_std', 'safe_prior_std']

PARAM_LABELS = {
    'n1_evidence_sd': 'Evidence SD₁',
    'n2_evidence_sd': 'Evidence SD₂',
    'risky_prior_mu': 'Risky prior μ',
    'safe_prior_mu':  'Safe prior μ',
    'risky_prior_std': 'Risky prior σ',
    'safe_prior_std':  'Safe prior σ',
}

# The two SI recovery panels: (out_name, source, calibrated_design).
PANELS = {
    'figure_s05_recovery': dict(source='data', calibrated=True),
    'figure_s07_recovery': dict(source='prior', calibrated=False),
}


# --------------------------------------------------------------------------- #
# Computation (mean over 10 sims of the generating x estimated correlation)
# --------------------------------------------------------------------------- #
def compute_recovery(source, calibrated, bids_folder, n_sims=10):
    """Mean generating(rows) x estimated(cols) Pearson correlation matrix.

    Exactly the notebook computation: per simulation, correlate the wide
    [mean, ground truth] table and take the ground-truth x mean block; average
    the per-simulation matrices."""
    td = op.join(bids_folder, RECOVERY_DIR)
    mats = []
    for ix in range(1, n_sims + 1):
        key = f'source-{source}' + ('_calibrateddesign' if calibrated else '') + f'_{ix}'
        gp = op.join(td, f'simulated_parameters_{key}.tsv')
        pf = op.join(td, f'posterior_samples_{key}.nc')
        if not (op.exists(gp) and op.exists(pf)):
            continue
        gen = pd.read_csv(gp, sep='\t', index_col=[0])
        gen.index.name = 'subject'
        gen.columns.name = 'parameter'
        est = az.from_netcdf(pf).posterior[PARAMS].to_dataframe()
        est.columns.name = 'parameter'
        wide = (est.groupby('subject').mean().stack().to_frame('mean')
                .join(gen.stack().to_frame('ground truth'))
                .unstack('parameter'))
        c = wide.corr().loc[['ground truth'], ['mean']]
        c = c.droplevel(0, axis=0).droplevel(0, axis=1).loc[PARAMS, PARAMS]
        mats.append(c)
    if not mats:
        raise FileNotFoundError(f'No recovery sims found for source={source}, '
                                f'calibrated={calibrated} in {td}')
    M = sum(m.values for m in mats) / len(mats)
    out = pd.DataFrame(M, index=PARAMS, columns=PARAMS)
    out.index.name = 'generating'
    out.columns.name = 'estimated'
    return out, len(mats)


def load_recovery(out_name, bids_folder, recompute=False):
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)
    tsv = op.join(sd_dir, f'{out_name}.tsv')
    spec = PANELS[out_name]
    if recompute or not op.exists(tsv):
        M, nsim = compute_recovery(spec['source'], spec['calibrated'], bids_folder)
        with open(tsv, 'w') as f:
            f.write(f'# Mean generating(rows) x estimated(cols) Pearson r over '
                    f'{nsim} simulations; source={spec["source"]}, '
                    f'calibrated_design={spec["calibrated"]}\n')
            M.to_csv(f, sep='\t')
        print(f'Wrote source data: {tsv}  (nsim = {nsim})')
    else:
        with open(tsv) as f:
            header = f.readline()
        nsim = int(header.split('over')[1].split('simulations')[0])
        M = pd.read_csv(tsv, sep='\t', comment='#', index_col=0).loc[PARAMS, PARAMS]
    return M, nsim


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(M, nsim):
    """Diverging (blue/red) recovery matrix, generating (rows) x estimated
    (cols); diagonal = recoverability."""
    import matplotlib.pyplot as plt
    style.set_style()

    A = M.loc[PARAMS, PARAMS].values.astype(float)
    k = len(PARAMS)
    VMIN, VMAX = -1.0, 1.0

    fig, ax = plt.subplots(figsize=(style.WIDTH_ONEHALF + 0.6, 4.5))
    fig.subplots_adjust(left=0.22, right=0.87, top=0.9, bottom=0.28)

    im = ax.imshow(A, cmap='RdBu_r', vmin=VMIN, vmax=VMAX, aspect='equal')

    for i in range(k):
        for j in range(k):
            r = A[i, j]
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

    ax.set_xlabel('Estimated', labelpad=6)
    ax.set_ylabel('Generating', labelpad=6)

    ax.text(1.0, -0.28, f'{nsim} simulations', transform=ax.transAxes,
            ha='right', va='top', fontsize=7.5, color='0.35')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label('Pearson r', labelpad=4)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(width=0.6, length=3)

    return fig


def main(bids_folder=BIDS, recompute=False):
    out_dir = style.figures_dir()
    os.makedirs(out_dir, exist_ok=True)
    for out_name in PANELS:
        M, nsim = load_recovery(out_name, bids_folder, recompute=recompute)
        fig = build_figure(M, nsim)
        style.save_panel(fig, out_name)
        fig.savefig(op.join(out_dir, f'{out_name}.png'), dpi=300)
        print(f'Wrote figure: {op.join(out_dir, out_name)}.pdf  '
              f'(diagonal {np.diag(M.values).min():.2f}-{np.diag(M.values).max():.2f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-folder', default=BIDS)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    main(args.bids_folder, recompute=args.recompute)
