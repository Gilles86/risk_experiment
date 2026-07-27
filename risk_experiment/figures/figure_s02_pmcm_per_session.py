"""Supplementary Figure 2 -- the per-session (3T / 7T) twin of Figure 2.

Main-text Figure 2 combines the 3T and 7T sessions (model-12). This
supplementary figure shows the *same* analysis performed separately for each
scanner, restyled to match Figure 2's house style. Layout is two scanner blocks
side-by-side (3T | 7T), each with three stacked rows:

  A  Posterior predictive of the full PMCM (per-session ``model-1``). Its
     asymmetric first/second-option evidence noise reproduces the order effect,
     so the prediction splits by order (blue = safe-first, orange = risky-first;
     coloured curves + 95% HDI bands). Two views: proportion-risky vs. binned
     risky/safe payoff and vs. the safe offer.
  B  Group-level parameter posteriors (Evidence SD, Prior mu, Prior std) from
     the per-session model-1 trace.
  C  Participant-level parameter differences (Option 1 - option 2; Risky - safe)
     coloured by each participant's per-session risk profile.

Corresponding panels (same panel type across the two scanners) share one common
x-range so the 3T and 7T columns line up exactly.

Data provenance (unchanged numbers): all rows use the per-session
``ses-{3t2,7t2}_model-1`` traces, exactly as the original figure2.ipynb did for
the separate-session panels.

The expensive step (build bauer model + ``model.ppc``) is cached per
(model, session) to a source-data TSV; restyling never re-samples. Pass
``--recompute`` to force it.

Usage:
    python -m risk_experiment.figures.figure_s02_pmcm_per_session
    python -m risk_experiment.figures.figure_s02_pmcm_per_session --recompute
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns

from risk_experiment.figures import style

BIDS = '/data/ds-risk'
SESSIONS = [('3t2', '3T'), ('7t2', '7T')]
ORDER_LABELS = {False: 'Safe first', True: 'Risky first'}
ORDER_HUE = ['Safe first', 'Risky first']

# Risk-category colours -- match the FINAL Figure 2 (red / teal / black).
RISK_COLORS = {'risk-seeking': style.RISKTYPE_COLORS['Risk-seeking'],
               'risk-averse': style.RISKTYPE_COLORS['Risk-averse'],
               'risk-neutral': style.RISKTYPE_COLORS['Risk-neutral']}
RISK_LABELS = {'risk-seeking': 'Risk-seeking', 'risk-averse': 'Risk-averse',
               'risk-neutral': 'Risk-neutral'}

# Group-posterior specs: (title, var_a, var_b, label_a, label_b, hue_order, palette)
GROUP_SPECS = [
    ('Evidence SD', 'n1_evidence_sd_mu', 'n2_evidence_sd_mu', 'Option 1', 'Option 2',
     ['Option 1', 'Option 2'], list(sns.color_palette('tab10')[4:6])),
    ('Prior mu', 'risky_prior_mu_mu', 'safe_prior_mu_mu', 'Risky', 'Safe',
     ['Safe', 'Risky'], [sns.color_palette('coolwarm', 4)[i] for i in (0, 3)]),
    ('Prior std', 'risky_prior_sd_mu', 'safe_prior_sd_mu', 'Risky', 'Safe',
     ['Safe', 'Risky'], [sns.color_palette('coolwarm', 4)[i] for i in (0, 3)]),
]

# Participant-difference specs: (title, x-label, var_a, var_b) -> a - b.
DIFF_SPECS = [
    ('Evidence SD (difference)', 'Option 1 - option 2', 'n1_evidence_sd', 'n2_evidence_sd'),
    ('Prior mu (difference)', 'Risky - safe', 'risky_prior_mu', 'safe_prior_mu'),
    ('Prior std (difference)', 'Risky - safe', 'risky_prior_sd', 'safe_prior_sd'),
]


# --------------------------------------------------------------------------- #
# Posterior-predictive computation (cached)
# --------------------------------------------------------------------------- #
def _hdi(x):
    lo, hi = az.hdi(np.asarray(x), hdi_prob=0.95)
    return pd.Series({'hdi_lo': lo, 'hdi_hi': hi})


def _ppc_raw(model_label, session, bids_folder):
    """Per-trial posterior-predictive (proportion chose risky), one session."""
    import pymc as pm
    style.shim_pymc_for_bauer(pm)
    from risk_experiment.cogmodels.fit_model import build_model, get_data

    if model_label == '1':                       # per-session fit
        df = get_data(model_label='1', session=session, bids_folder=bids_folder, roi=None)
        trace = op.join(bids_folder, 'derivatives', 'cogmodels',
                        f'ses-{session}_model-1_trace.netcdf')
    else:                                        # combined klw, split by session
        df = get_data(model_label=model_label, session=None, bids_folder=bids_folder, roi=None)
        trace = op.join(bids_folder, 'derivatives', 'cogmodels',
                        f'model-{model_label}_trace.netcdf')
    model = build_model(model_label=model_label, df=df, roi=None)
    idata = az.from_netcdf(trace)

    paradigm = df.drop('session', axis=1) if 'session' in df.columns else df
    ppc = model.ppc(idata=idata.sel(draw=slice(None, None, 20)),
                    paradigm=paradigm, var_names=['ll_bernoulli'])
    ppc = ppc.stack([0, 1]).to_frame('choice_pred')
    ppc = ppc.reset_index().set_index(df.index.names + ['chain', 'draw'])[['choice_pred']]
    ppc = ppc.join(df[['risky_first', 'n_risky', 'n_safe', 'chose_risky']]).reset_index()
    if 'session' in ppc.columns:
        ppc = ppc[ppc['session'] == session]
    # flip so 1 == chose risky irrespective of presentation order
    ppc['chose_risky_pred'] = np.where(ppc['risky_first'],
                                       1 - ppc['choice_pred'], ppc['choice_pred'])
    ppc['ratio_bin'] = ppc.groupby('subject').apply(
        lambda x: pd.qcut(x['n_risky'] / x['n_safe'], 7, labels=False,
                          duplicates='drop')).reset_index(level=0, drop=True)
    return ppc


def _agg(ppc, xcol, by_order):
    keys = (['risky_first', xcol] if by_order else [xcol])
    tmp = ppc.groupby(keys + ['chain', 'draw'])[['chose_risky_pred', 'chose_risky']].mean()
    grp = tmp.reset_index().groupby(keys)
    out = grp['chose_risky_pred'].mean().rename('pred_mean').to_frame()
    out = out.join(grp['chose_risky_pred'].apply(lambda z: _hdi(z)).unstack())
    out = out.join(grp['chose_risky'].mean().rename('data_mean'))
    return out.reset_index()


def compute_ppc(model_label, session, bids_folder):
    """Tidy PPC table: one row per (x_kind, x, order)."""
    ppc = _ppc_raw(model_label, session, bids_folder)
    rows = []
    for x_kind, xcol in [('ratio', 'ratio_bin'), ('safe', 'n_safe')]:
        pooled = _agg(ppc, xcol, by_order=False).rename(columns={xcol: 'x'})
        pooled['order'] = 'pooled'
        rows.append(pooled)
        byo = _agg(ppc, xcol, by_order=True).rename(columns={xcol: 'x'})
        byo['order'] = byo['risky_first'].map(ORDER_LABELS)
        rows.append(byo.drop(columns='risky_first'))
        for r in rows[-2:]:
            r['x_kind'] = x_kind
    out = pd.concat(rows, ignore_index=True)
    out['model'] = model_label
    out['session'] = session
    return out[['model', 'session', 'x_kind', 'x', 'order',
                'pred_mean', 'hdi_lo', 'hdi_hi', 'data_mean']]


def load_ppc(bids_folder, recompute=False):
    sd_dir = style.source_data_dir(bids_folder)
    os.makedirs(sd_dir, exist_ok=True)
    tsv = op.join(sd_dir, 'figure_s02AB_ppc.tsv')
    if recompute or not op.exists(tsv):
        frames = []
        for model_label in ['1']:
            for session, _ in SESSIONS:
                print(f'Computing PPC: model-{model_label}, ses-{session} ...')
                frames.append(compute_ppc(model_label, session, bids_folder))
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(tsv, sep='\t', index=False)
        print(f'Wrote source data: {tsv}')
    else:
        out = pd.read_csv(tsv, sep='\t')
    return out


# --------------------------------------------------------------------------- #
# Group + participant posteriors (cheap; read straight from the netcdf)
# --------------------------------------------------------------------------- #
def compute_group(bids_folder):
    frames = []
    for session, _ in SESSIONS:
        idata = az.from_netcdf(op.join(bids_folder, 'derivatives', 'cogmodels',
                                       f'ses-{session}_model-1_trace.netcdf'))
        for title, va, vb, la, lb, _, _ in GROUP_SPECS:
            a = idata.posterior[va].values.ravel()
            b = idata.posterior[vb].values.ravel()
            frames.append(pd.DataFrame({'session': session, 'panel': title,
                                        la: a, lb: b})
                          .melt(id_vars=['session', 'panel'],
                                var_name='condition', value_name='value'))
    return pd.concat(frames, ignore_index=True)


def compute_diffs(bids_folder):
    rp = pd.read_csv(op.join(bids_folder, 'derivatives', 'cogmodels',
                             'simple_risk_preference.tsv'),
                     index_col=[0, 1], sep='\t', dtype={'subject': str})
    out = []
    for session, _ in SESSIONS:
        idata = az.from_netcdf(op.join(bids_folder, 'derivatives', 'cogmodels',
                                       f'ses-{session}_model-1_trace.netcdf'))
        rp_s = rp.xs(session, 0, 'session')
        for title, _xlabel, va, vb in DIFF_SPECS:
            diff = (idata.posterior[va] - idata.posterior[vb]).to_dataframe(
                name='diff')
            hdi = (diff.groupby('subject')['diff']
                   .apply(lambda d: pd.Series(az.hdi(d.values),
                                              index=['low', 'high'])).unstack())
            mean = diff.groupby('subject')['diff'].mean().rename('mean')
            tmp = mean.to_frame().join(hdi).join(rp_s)
            tmp['session'] = session
            tmp['panel'] = title
            out.append(tmp.reset_index())
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _despine(ax, left=True):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines['left'].set_visible(False)


def _unify_x(axes, pad_frac=0.03):
    """Force a set of corresponding axes (same panel across scanners) to share
    one common x-range, so 3T and 7T columns line up exactly."""
    lo = min(ax.get_xlim()[0] for ax in axes)
    hi = max(ax.get_xlim()[1] for ax in axes)
    pad = (hi - lo) * pad_frac
    for ax in axes:
        ax.set_xlim(lo - pad, hi + pad)


def build_figure(ppc, group, diffs, bids_folder):
    import matplotlib.pyplot as plt
    style.set_style()

    xpos, xlab = style.ratio_bin_ticks(bids_folder)
    x_neutral = style.ratio_pos(1 / 0.55, bids_folder)
    PPC_YLIM = (0.13, 0.87)
    PPC_YTICKS = [0.2, 0.4, 0.6, 0.8]
    RATIO_XLIM = (-0.4, 6.4)     # binned 0..6 index, shared across scanners
    SAFE_XLIM = (3.5, 29.5)      # safe offer 5..28, shared across scanners

    # Overall proportions chosen to match the target mock-up (~1.6:1 wide),
    # with a shorter group-posterior (KDE) row than the PPC / forest rows.
    fig = plt.figure(figsize=(9.6, 6.1))
    outer = fig.add_gridspec(3, 2, width_ratios=[1, 1],
                             height_ratios=[1.2, 0.8, 1.25],
                             left=0.065, right=0.987, top=0.9, bottom=0.075,
                             hspace=0.62, wspace=0.14)

    # collect corresponding axes (col index -> [3T ax, 7T ax]) to unify x later
    xcols = {'ppc': {}, 'group': {}, 'diff': {}}

    def ppc_row(row, model_label):
        for si, (session, _tag) in enumerate(SESSIONS):
            sub = outer[row, si].subgridspec(1, 2, wspace=0.34)
            axes = [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])]
            axes[1].sharey(axes[0])
            for ci, (ax, (x_kind, xlabel)) in enumerate(
                    zip(axes, [('ratio', 'Risky / safe payoff'),
                               ('safe', 'Safe offer')])):
                cell = ppc[(ppc['model'] == model_label) & (ppc['session'] == session)
                           & (ppc['x_kind'] == x_kind)]
                ax.axhline(0.5, c='k', ls='--', lw=0.6, zorder=0)
                if x_kind == 'ratio':
                    ax.axvline(x_neutral, c='k', ls='--', lw=0.6, zorder=0)
                for order in ORDER_HUE:
                    pred = cell[cell['order'] == order].sort_values('x')
                    c = style.ORDER_COLORS[order]
                    ax.fill_between(pred['x'], pred['hdi_lo'], pred['hdi_hi'],
                                    color=c, alpha=0.2, lw=0, zorder=1)
                    ax.plot(pred['x'], pred['pred_mean'], color=c, lw=1.7, zorder=2)
                for order in ORDER_HUE:
                    d = cell[cell['order'] == order].sort_values('x')
                    ax.plot(d['x'], d['data_mean'], marker='o', ls='', ms=4,
                            color=style.ORDER_COLORS[order], mec='none', zorder=4)
                ax.set_ylim(*PPC_YLIM)
                ax.set_yticks(PPC_YTICKS)
                ax.set_xlabel(xlabel)
                if x_kind == 'ratio':
                    ax.set_xticks(xpos)
                    ax.set_xticklabels(xlab)
                    ax.set_xlim(*RATIO_XLIM)
                else:
                    ax.set_xticks([10, 20])
                    ax.set_xlim(*SAFE_XLIM)
                _despine(ax)
                if x_kind == 'safe':
                    ax.tick_params(labelleft=False)
                xcols['ppc'].setdefault(ci, []).append(ax)
            axes[0].set_ylabel('P(risky choice)')
            # order legend once, in the rightmost panel of the row (7T safe offer)
            if si == len(SESSIONS) - 1:
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], marker='o', ls='none', ms=5,
                                  color=style.ORDER_COLORS[o]) for o in ORDER_HUE]
                leg = axes[1].legend(handles, ORDER_HUE, loc='upper right',
                                     frameon=True, fontsize=8, handletextpad=0.3,
                                     borderaxespad=0.4, edgecolor='0.6',
                                     facecolor='white', framealpha=1.0)
                leg.get_frame().set_linewidth(0.6)

    def group_row(row):
        for si, (session, _tag) in enumerate(SESSIONS):
            sub = outer[row, si].subgridspec(1, 3, wspace=0.42)
            for ci, (title, va, vb, la, lb, hue_order, palette) in enumerate(GROUP_SPECS):
                ax = fig.add_subplot(sub[0, ci])
                cmap = dict(zip(hue_order, palette))
                cell = group[(group['session'] == session) & (group['panel'] == title)]
                for cond in hue_order:
                    vals = cell[cell['condition'] == cond]['value'].values
                    sns.kdeplot(x=vals, fill=True, color=cmap[cond], alpha=0.5,
                                lw=1.0, ax=ax, label=cond)
                ax.set_title(title)
                ax.set_xlabel('')
                ax.set_ylabel('Posterior density' if ci == 0 else '')
                ax.set_yticks([])
                _despine(ax)
                ax.tick_params(axis='y', length=0)
                if si == 0:                       # legend only on the 3T block
                    leg = ax.legend(frameon=True, fontsize=8, handlelength=1.0,
                                    handletextpad=0.4, edgecolor='0.6',
                                    facecolor='white', framealpha=1.0)
                    leg.get_frame().set_linewidth(0.6)
                else:
                    lg = ax.get_legend()
                    if lg:
                        lg.remove()
                xcols['group'].setdefault(ci, []).append(ax)

    def diff_row(row):
        for si, (session, _tag) in enumerate(SESSIONS):
            sub = outer[row, si].subgridspec(1, 3, wspace=0.42)
            for ci, (title, xlabel, _, _) in enumerate(DIFF_SPECS):
                ax = fig.add_subplot(sub[0, ci])
                cell = diffs[(diffs['session'] == session) & (diffs['panel'] == title)].copy()
                order = cell.sort_values('mean')['subject'].tolist()
                ax.axvline(0, c='k', ls='--', lw=0.6, zorder=0)
                for _, r in cell.iterrows():
                    y = -order.index(r['subject'])
                    ax.plot([r['low'], r['high']], [y, y], color='0.6', lw=2.0, zorder=2)
                    ax.scatter([r['mean']], [y], s=16,
                               color=RISK_COLORS[r['risk_profile']], zorder=5)
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_yticks([])
                ax.set_ylabel('Participant' if ci == 0 else '')
                _despine(ax)
                ax.tick_params(axis='y', length=0)
                xcols['diff'].setdefault(ci, []).append(ax)
            # risk-type legend once, on the 7T last forest panel
            if si == 1:
                from matplotlib.lines import Line2D
                keys = ['risk-seeking', 'risk-averse', 'risk-neutral']
                handles = [Line2D([0], [0], marker='o', ls='none', ms=5,
                                  color=RISK_COLORS[k], label=RISK_LABELS[k])
                           for k in keys]
                leg = ax.legend(handles=handles, loc='lower right', frameon=True,
                                fontsize=8, handletextpad=0.3, labelspacing=0.35,
                                edgecolor='0.6', facecolor='white', framealpha=1.0)
                leg.get_frame().set_linewidth(0.6)

    ppc_row(0, '1')
    group_row(1)
    diff_row(2)

    # corresponding panels (same column, both scanners) share one x-range
    for kind in ('group', 'diff'):
        for axlist in xcols[kind].values():
            _unify_x(axlist)

    # scanner headers (centred over each scanner column)
    for si, (_session, tag) in enumerate(SESSIONS):
        bb = outer[0, si].get_position(fig)
        fig.text((bb.x0 + bb.x1) / 2, 0.935, tag, ha='center', va='center',
                 fontsize=13, **style.BOLD)

    # panel letters at the top-left of each row
    for row, letter in enumerate(['A', 'B', 'C']):
        bb = outer[row, 0].get_position(fig)
        fig.text(0.012, bb.y1 + 0.01, letter, ha='left', va='bottom',
                 fontsize=13, **style.BOLD)

    return fig


def main(bids_folder=BIDS, recompute=False):
    import matplotlib.pyplot as plt
    ppc = load_ppc(bids_folder, recompute=recompute)
    group = compute_group(bids_folder)
    diffs = compute_diffs(bids_folder)

    fig = build_figure(ppc, group, diffs, bids_folder)

    out_dir = style.figures_dir()
    os.makedirs(out_dir, exist_ok=True)
    pdf = op.join(out_dir, 'figure_s02_pmcm_per_session.pdf')
    png = op.join(out_dir, 'figure_s02_pmcm_per_session.png')
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(f'Wrote figure: {pdf}')
    print(f'Wrote figure: {png}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-folder', default=BIDS)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    main(args.bids_folder, recompute=args.recompute)
