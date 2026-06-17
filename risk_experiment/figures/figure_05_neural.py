"""Figure 5 (the old neural Figure 4) -- nPRF decoding + neural-uncertainty effects.

New Figure 5 == old Figure 4 (see FIGURE_REGENERATION_BRIEF.md section 0). It
has four panels:

  * 5A  nPRF preferred-numerosity surface maps (3T & 7T).  These are pycortex
        BRAIN RENDERS (raster).  They are *not* produced here -- the user
        re-exports them manually and composites them in Affinity.  This script
        only handles the data-driven panels 5B-5D.

  * 5B  Decoding scatters/swarms: per-participant correlation of (i) decoded
        posterior mean (E) vs objective numerosity log(n1), and (ii) decoded
        posterior sd vs absolute decoding error.  x = Scanner (3T / 7T).
        Reproduces analyze_decoding_natural_space.ipynb (decoding_r* pdfs).

  * 5C  Distance-from-risk-neutral posterior densities, low vs high neural
        uncertainty (combined 3T+7T probit_neural9 model).  Reproduces
        analyze_neural_probit.ipynb cell 28 (model-rnp_neural9_rnp_distance.pdf).

  * 5D  Group-level posteriors of the *neural* regressor (the effect of decoded
        numerosity sd) on the model parameters: Evidence sd, Prior mean,
        Prior std, hue = option (Safe / Risky).  Reproduces the neural
        RiskRegressionModel (model neural32) group coefficients in the
        figure-2 group-posterior style.

The expensive steps (building the bauer / bambi models + extracting fake-data
predictions, loading every decoded pdf) are cached to per-panel source-data
TSVs.  Re-styling never re-runs them.  Pass --recompute to force.

This is a FORMATTING pass: numbers and statistics are reproduced unchanged from
the published analysis; only the rendering follows the house style.

Usage
-----
    python -m risk_experiment.figures.figure_05_neural
    python -m risk_experiment.figures.figure_05_neural --recompute
"""
import argparse
import os
import os.path as op

import numpy as np
import pandas as pd
import seaborn as sns

from risk_experiment.figures import style


BIDS_DEFAULT = '/data/ds-risk'

# Semantic colours kept from the source notebooks.
# Low / High neural uncertainty == seaborn tab10 indices 2 (green) and 3 (red),
# exactly as the notebooks use `sns.color_palette()[2:]`.
NEURAL_UNCERTAINTY_COLORS = {
    'Low neural uncertainty': sns.color_palette('tab10')[2],
    'High neural uncertainty': sns.color_palette('tab10')[3],
}
NEURAL_UNCERTAINTY_ORDER = ['Low neural uncertainty', 'High neural uncertainty']

# Option Safe / Risky == coolwarm endpoints, exactly as figure2.ipynb.
_coolwarm4 = sns.color_palette('coolwarm', 4)
OPTION_COLORS = {'Safe': _coolwarm4[0], 'Risky': _coolwarm4[3]}
OPTION_ORDER = ['Safe', 'Risky']

# Per-panel hue scheme (matches Figure 2B): the evidence-noise panel contrasts
# the first vs second presented option (Option 1/2, purple/brown = tab10[4:6]);
# the prior panels contrast the safe vs risky option (Safe/Risky, coolwarm).
EVIDENCE_COLORS = {'Option 1': sns.color_palette('tab10')[4],
                   'Option 2': sns.color_palette('tab10')[5]}
PANEL_HUE = {
    'Evidence sd': (['Option 1', 'Option 2'], EVIDENCE_COLORS),
    'Prior mean': (OPTION_ORDER, OPTION_COLORS),
    'Prior std': (OPTION_ORDER, OPTION_COLORS),
}


# ---------------------------------------------------------------------------
# 5B  decoding correlations
# ---------------------------------------------------------------------------

def compute_decoding(bids_folder=BIDS_DEFAULT):
    """Per-participant decoding correlations, reproducing the decoding notebook.

    Returns a tidy DataFrame with columns:
      subject, session, Scanner, r_E_n1 (corr E vs log(n1)),
      r_sd_error (corr sd vs |error|).
    """
    import pingouin
    from risk_experiment.utils.data import get_all_subjects

    # Mirrors revision/report_statistics.py (the audited source of truth):
    #  accuracy r  = per-run corr(E, log(n1)) averaged over runs, per subject;
    #  uncertainty r = pooled corr(sd, |n1 - E|), per subject.
    # Extract the scalar r (pingouin.corr returns a full frame -- averaging the
    # whole frame is what crashed before on its string/array columns).
    rows = []
    for sub in get_all_subjects(bids_folder):
        for ses in ['3t2', '7t2']:
            try:
                pred = sub.get_decoding_info(ses, mask='npcr', n_voxels=0.0)
                beh = sub.get_behavior(sessions=ses)
                m = pred.join(beh[['n1']]).dropna(subset=['n1', 'E'])
                if len(m) <= 10:
                    continue
                m['log(n1)'] = np.log(m['n1'])
                if 'run' in m.index.names:
                    r_acc = m.groupby('run').apply(
                        lambda x: pingouin.corr(x['E'], x['log(n1)'])['r'].iloc[0]
                    ).mean()
                else:
                    r_acc = pingouin.corr(m['E'], m['log(n1)'])['r'].iloc[0]
                r_err = pingouin.corr(m['sd'], np.abs(m['n1'] - m['E']))['r'].iloc[0]
                rows.append({'subject': sub.subject, 'session': ses,
                             'Scanner': {'3t2': '3T', '7t2': '7T'}[ses],
                             'r_E_n1': r_acc, 'r_sd_error': r_err})
            except Exception:
                pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5C  distance to risk-neutral, low vs high neural uncertainty (probit_neural9)
# ---------------------------------------------------------------------------

def compute_rnp_distance(bids_folder=BIDS_DEFAULT):
    """Reproduce analyze_neural_probit.ipynb cell 28 (both sessions, model 9).

    Returns a long DataFrame of the per-(chain, draw) group-mean distance to
    risk-neutral, one row per posterior sample, with a 'Neural uncertainty'
    column.
    """
    import arviz as az
    from risk_experiment.cogmodels.fit_probit import build_model, get_data
    from risk_experiment.cogmodels.utils import extract_intercept_gamma

    model_label = 'probit_neural9'
    df = get_data(model_label, None, bids_folder)
    model = build_model(model_label, df, None, bids_folder)
    idata = az.from_netcdf(
        op.join(bids_folder, 'derivatives', 'cogmodels',
                f'model-{model_label}_trace.netcdf'))

    intercept, gamma = extract_intercept_gamma(idata, model, df, group=False)
    rnp = np.exp(intercept['intercept'] / gamma['gamma']).stack([-2, -1]).to_frame('rnp')
    rnp = rnp[(rnp > 0.0) & (rnp < 1.)]

    rnp['distance_to_risk_neutral'] = (rnp['rnp'] - .55).abs()
    rnp['Neural uncertainty'] = rnp.index.get_level_values('median_split_sd').map(
        {True: 'High neural uncertainty', False: 'Low neural uncertainty'})

    tmp = rnp.groupby(['chain', 'draw', 'Neural uncertainty']).mean()
    out = tmp.reset_index()[['chain', 'draw', 'Neural uncertainty',
                             'distance_to_risk_neutral']]
    return out


# ---------------------------------------------------------------------------
# 5D  neural regressor group posteriors (Evidence sd / Prior mean / Prior std)
# ---------------------------------------------------------------------------

# Which group-level coefficient maps to which (Parameter, Option) cell, and the
# design-matrix column that carries the decoded-sd ("neural uncertainty")
# regressor for the neural32 model (`sd*session` -> columns 'Intercept', 'sd',
# 'session[...]', 'sd:session[...]'; the main neural effect is column 'sd').
NEURAL_REGRESSOR = 'sd'
NEURAL_PARAM_MAP = [
    # (trace variable,            Parameter label, Option)
    ('n1_evidence_sd_mu', 'Evidence sd', 'Option 1'),
    ('n2_evidence_sd_mu', 'Evidence sd', 'Option 2'),
    ('risky_prior_mu_mu', 'Prior mean', 'Risky'),
    ('safe_prior_mu_mu',  'Prior mean', 'Safe'),
    ('risky_prior_sd_mu', 'Prior std',  'Risky'),
    ('safe_prior_sd_mu',  'Prior std',  'Safe'),
]


# Which panel annotates which coefficient's p(slope<0) (the manuscript-reported
# neural effects). Prior mean has no significant neural effect -> not annotated.
REPORTED_EFFECT = {'Evidence sd': 'Option 1', 'Prior std': 'Safe'}


def compute_neural_gamma(bids_folder=BIDS_DEFAULT, model_label='neural32'):
    """Group posteriors of the decoded-sd regressor coefficient per parameter.

    Reproduces the neural RiskRegressionModel group coefficients shown in the
    old Figure 4D: for each model parameter (n1/n2 evidence sd, risky/safe prior
    mu, risky/safe prior sd) we extract the *neural-uncertainty* (decoded sd)
    regressor coefficient's group-level posterior (`<param>_mu` along the
    `<param>_regressors == 'sd'` column).
    """
    import arviz as az

    idata = az.from_netcdf(
        op.join(bids_folder, 'derivatives', 'cogmodels',
                f'model-{model_label}_trace.netcdf'))

    records = []
    for var, param, option in NEURAL_PARAM_MAP:
        if var not in idata.posterior:
            raise KeyError(f'{var} not in trace for model {model_label}')
        da = idata.posterior[var]
        reg_dim = [d for d in da.dims if d.endswith('_regressors')][0]
        # Combined (session-averaged) decoded-sd slope, exactly as the audited
        # report_statistics.py: sd + 0.5 * sd:session[T.7t2]. This reproduces
        # the manuscript p-values (v1: 0.009, safe prior SD: 0.004).
        val = (da.sel({reg_dim: 'sd'})
               + 0.5 * da.sel({reg_dim: 'sd:session[T.7t2]'}))
        s = val.to_dataframe()[var]
        rec = s.reset_index()[['chain', 'draw', var]].rename(columns={var: 'value'})
        rec['Parameter'] = param
        rec['Option'] = option
        records.append(rec)

    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

def _load_or_compute(name, fn, bids_folder, recompute):
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)
    tsv = op.join(sd_dir, f'{name}.tsv')
    if recompute or not op.exists(tsv):
        out = fn(bids_folder)
        out.to_csv(tsv, sep='\t', index=False)
        print(f'Wrote source data: {tsv}')
    else:
        out = pd.read_csv(tsv, sep='\t', dtype={'subject': str})
        print(f'Loaded cached source data: {tsv}')
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_decoding(dec, bids_folder=BIDS_DEFAULT):
    """5B: two swarmplots (r values) x = Scanner, dashed zero line."""
    import matplotlib.pyplot as plt

    style.set_style()

    # Width: decoding (2 subpanels) + neural_gamma (3 subpanels) share the
    # 180 mm bottom row, so size them to ~equal subpanel width and sum < 180.
    # Wider left panel; height matched to neural_gamma (5D) so the row aligns.
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_DOUBLE * 0.50, 1.7),
                             constrained_layout=True)

    # Short one-line titles; the y-axis ("Correlation (r)") already says these
    # are correlations, so the title only needs to name the two quantities.
    specs = [
        ('r_E_n1', 'Decoded mean vs numerosity'),
        ('r_sd_error', 'Decoded SD vs error'),
    ]
    order = ['3T', '7T']
    for ax, (col, title) in zip(axes, specs):
        ax.axhline(0.0, c='k', ls='--', lw=0.8, zorder=0)
        # Translucent per-participant swarm (smaller -> narrower cloud) ...
        sns.swarmplot(data=dec, x='Scanner', y=col, order=order,
                      color='0.6', size=2.6, alpha=0.55, ax=ax, zorder=2)
        # ... with the mean +/- SEM diamond overlaid on the cloud. The SEM is
        # genuinely small (~0.02-0.03 r); prominent caps let the short bar peek
        # above/below the diamond.
        for k, scanner in enumerate(order):
            vals = dec.loc[dec['Scanner'] == scanner, col].dropna().values
            m = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            ax.errorbar(k, m, yerr=se, fmt='D', color='k', mfc='white',
                        ms=6, mew=1.5, elinewidth=1.8, capsize=4, capthick=1.5,
                        zorder=10)
        ax.set_title(title, fontsize=7)
        # y-label once (left panel); both panels are "Correlation (r)".
        ax.set_ylabel('Correlation (r)' if ax is axes[0] else '')
        ax.set_xlabel('Scanner')
        # Pull the two categories toward the panel centre (wider x-limits),
        # leaving room on the right of each cloud for the mean +/- SEM marker.
        ax.set_xlim(-0.9, 2.1)
        sns.despine(ax=ax, offset=4, trim=False)

    pdf = style.save_panel(fig, 'figure_05_decoding', bids_folder)
    print(f'Wrote figure: {pdf}')
    return fig


def plot_rnp_distance(dist, bids_folder=BIDS_DEFAULT):
    """5C: KDE of group-mean distance to risk-neutral, low vs high uncertainty."""
    import matplotlib.pyplot as plt

    style.set_style()

    # Width matched to neural_gamma (5D) so the right column (5C over 5D) aligns.
    fig, ax = plt.subplots(figsize=(style.WIDTH_DOUBLE * 0.46, 1.7),
                           constrained_layout=True)

    for label in NEURAL_UNCERTAINTY_ORDER:
        sub = dist[dist['Neural uncertainty'] == label]
        sns.kdeplot(data=sub, x='distance_to_risk_neutral', fill=True,
                    color=NEURAL_UNCERTAINTY_COLORS[label], label=label, ax=ax,
                    lw=1.0)

    # Posterior probability that low < high uncertainty distance, exactly the
    # quantity the notebook prints (`(split[False] > split[True]).mean()`).
    wide = dist.pivot_table(index=['chain', 'draw'], columns='Neural uncertainty',
                            values='distance_to_risk_neutral')
    p = (wide['Low neural uncertainty'] > wide['High neural uncertainty']).mean()
    ann = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    ax.annotate(ann, xy=(0.02, 0.97), xycoords='axes fraction', ha='left',
                va='top', fontsize=8)

    ax.set_xlabel('Distance from risk-neutrality')
    ax.set_ylabel(None)
    ax.set_yticks([])
    # Headroom so the legend + p-value clear the distributions.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    leg = ax.legend(loc='upper right', frameon=True, fontsize=8, handlelength=1.0,
                    handletextpad=0.4, edgecolor='0.6', facecolor='white',
                    framealpha=1.0)
    leg.get_frame().set_linewidth(0.6)
    sns.despine(ax=ax, offset=4, trim=False, left=True)
    ax.tick_params(axis='y', length=0)

    pdf = style.save_panel(fig, 'figure_05_rnp_distance', bids_folder)
    print(f'Wrote figure: {pdf}')
    return fig


def plot_neural_gamma(gam, bids_folder=BIDS_DEFAULT):
    """5D: group posteriors of the decoded-sd regressor coefficient.

    One sub-panel per parameter (Evidence sd, Prior mean, Prior std); within
    each, hue = Option (Safe / Risky); dashed zero reference.
    """
    import matplotlib.pyplot as plt

    style.set_style()

    params = ['Evidence sd', 'Prior mean', 'Prior std']
    # Right column ~20% narrower than before, so the left panels read wider.
    fig, axes = plt.subplots(1, len(params),
                             figsize=(style.WIDTH_DOUBLE * 0.46, 1.7),
                             constrained_layout=True)

    for ax, param in zip(axes, params):
        sub = gam[gam['Parameter'] == param]
        hue_order, colors = PANEL_HUE[param]
        for option in hue_order:
            s = sub[sub['Option'] == option]
            if s.empty:
                continue
            sns.kdeplot(data=s, x='value', fill=True,
                        color=colors[option], label=option, ax=ax, lw=1.0)
        # Stop the zero line below the top annotation band so it never crosses
        # the p-value / legend (in Prior std the data sit right of 0, putting the
        # line under the top-left p-value otherwise).
        ax.axvline(0.0, ymax=0.72, c='k', ls='--', lw=0.8, zorder=0)

        # Annotate the manuscript-reported neural effect for this parameter:
        # p(slope < 0) for the significant coefficient -- Evidence sd carries the
        # noisier-first-option effect (Option 1/v1, p=0.009); Prior std carries
        # the safe-payoff prior-width effect (Safe, p=0.004). Computed from the
        # posterior so the value stays faithful (not hard-coded).
        if param in REPORTED_EFFECT:
            opt = REPORTED_EFFECT[param]
            vals = sub[sub['Option'] == opt]['value'].values
            p = float((vals < 0).mean())
            ann = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
            # Sit the p-value just above the distribution it describes (its
            # median x), colour-matched -- the legends now hold the top corners.
            ax.text(float(np.median(vals)), 0.50, ann,
                    transform=ax.get_xaxis_transform(), ha='left', va='bottom',
                    fontsize=8, color=colors[opt])

        ax.set_title(param, fontsize=7)   # match 5B decoding titles (7 pt)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_yticks([])
        # Sparse, compact x-ticks: the narrow panels can't fit 3 long labels
        # (Prior mean's symmetric ±0.025 collide). Few ticks + stripped zeros.
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: '0' if abs(x) < 1e-9
            else f'{x:.3f}'.rstrip('0').replace('-0.', '-.').replace('0.', '.')))
        # Headroom so the top-right legend and the on-distribution p-value both
        # clear the KDE peaks in the narrow panels.
        ax.set_ylim(0, ax.get_ylim()[1] * 2.3)
        # Boxed key at the top-right (nudged to the edge) on both Evidence sd
        # (Option 1/2) and Prior mean (the single Safe/Risky key); Prior std
        # reuses the Prior-mean key. The p-values now live on the distributions,
        # so the top-right corner is free for the legend.
        if param in ('Evidence sd', 'Prior mean'):
            leg = ax.legend(loc='upper right', frameon=True, fontsize=8,
                            handlelength=1.0, handletextpad=0.4, borderaxespad=0.2,
                            edgecolor='0.6', facecolor='white', framealpha=1.0)
            leg.get_frame().set_linewidth(0.6)
        sns.despine(ax=ax, offset=4, trim=False, left=True)
        ax.tick_params(axis='y', length=0)

    pdf = style.save_panel(fig, 'figure_05_neural_gamma', bids_folder)
    print(f'Wrote figure: {pdf}')
    return fig


# ---------------------------------------------------------------------------

def main(bids_folder=BIDS_DEFAULT, recompute=False):
    dec = _load_or_compute('figure_05_decoding',
                           compute_decoding, bids_folder, recompute)
    dist = _load_or_compute('figure_05_rnp_distance',
                            compute_rnp_distance, bids_folder, recompute)
    gam = _load_or_compute('figure_05_neural_gamma',
                           compute_neural_gamma, bids_folder, recompute)

    plot_decoding(dec, bids_folder)
    plot_rnp_distance(dist, bids_folder)
    plot_neural_gamma(gam, bids_folder)

    print('\n5A (nPRF surface maps) is a manual pycortex render -- not produced '
          'here; the user re-exports and composites it in Affinity.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-folder', default=BIDS_DEFAULT)
    parser.add_argument('--recompute', action='store_true',
                        help='Force recomputation of the cached source data.')
    args = parser.parse_args()
    main(args.bids_folder, recompute=args.recompute)
