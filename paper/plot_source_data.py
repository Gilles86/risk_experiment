#!/usr/bin/env python3
"""
Reproduce paper figures from Source_Data.xlsx to verify correctness.
Run from repo root: python paper/plot_source_data.py
Output: paper/source_data_figures.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import os.path as op

SOURCE = op.join(op.dirname(op.abspath(__file__)), 'Source_Data.xlsx')
OUTPUT = op.join(op.dirname(op.abspath(__file__)), 'source_data_figures.pdf')

xl = pd.ExcelFile(SOURCE)

# ── colour / style helpers ────────────────────────────────────────────────────
ORDER_COLORS = {'Risky first': '#E6531D', 'Safe first': '#3B82C4'}
STAKE_COLORS = {'Small (5-7)': '#2196F3', 'Medium (10-14)': '#4CAF50',
                'Large (20-28)': '#FF5722'}
SCANNER_COLORS = {'3T': '#5C85D6', '7T': '#D65C5C'}

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 120,
})


def errband(ax, x, y, lo, hi, color, alpha=0.25):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def finish(ax, xlabel=None, ylabel=None, title=None, ylim=None):
    ax.spines[['top', 'right']].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight='bold')
    if ylim:
        ax.set_ylim(ylim)


# ── load all sheets ───────────────────────────────────────────────────────────
sheets = {name: xl.parse(name) for name in xl.sheet_names}

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
    ax.set_xlabel('Safe payoff (CHF)')
    ax.set_ylabel('Proportion chose risky')
    ax.legend(frameon=False)
    finish(ax, title='Figure 1B')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 1C ─────────────────────────────────────────────────────────────
    df = sheets['Figure 1C']
    rnp = sheets['Figure 1C RNP insets']
    stake_bins = ['Small (5-7)', 'Medium (10-14)', 'Large (20-28)']
    orders = ['Risky first', 'Safe first']

    fig, axes = plt.subplots(3, 2, figsize=(7, 7), sharey=True, sharex=True)
    for ri, stake in enumerate(stake_bins):
        for ci, order in enumerate(orders):
            ax = axes[ri, ci]
            grp = df[(df['stake_bin'] == stake) & (df['order'] == order)].sort_values('log_risky_safe_ratio_bin')
            ax.errorbar(grp['log_risky_safe_ratio_bin'], grp['mean'], yerr=grp['sem'],
                        color=ORDER_COLORS[order], marker='o', markersize=3,
                        linewidth=1.5, capsize=2)
            ax.axhline(0.5, color='k', linestyle='--', linewidth=0.7, alpha=0.5)

            # RNP inset text
            r = rnp[(rnp['stake_bin'] == stake) & (rnp['order'] == order)]
            if len(r):
                rnp_val = r['mean'].values[0]
                ax.text(0.05, 0.92, f'RNP={rnp_val:.2f}', transform=ax.transAxes,
                        fontsize=7, color=ORDER_COLORS[order])

            if ri == 0:
                ax.set_title(order)
            if ci == 0:
                ax.set_ylabel(f'{stake}\nProp. risky')
            finish(ax)

    axes[-1, 0].set_xlabel('log(risky/safe)')
    axes[-1, 1].set_xlabel('log(risky/safe)')
    fig.suptitle('Figure 1C', fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 2A ─────────────────────────────────────────────────────────────
    df = sheets['Figure 2A']
    fig, axes = plt.subplots(3, 2, figsize=(7, 7), sharey=True, sharex=True)
    for ri, stake in enumerate(stake_bins):
        for ci, order in enumerate(orders):
            ax = axes[ri, ci]
            grp = df[(df['stake_bin'] == stake) & (df['order'] == order)].sort_values('bin(risky/safe)')
            color = ORDER_COLORS[order]
            # model band
            errband(ax, grp['bin(risky/safe)'], grp['model_mean'],
                    grp['model_hdi_2_5'], grp['model_hdi_97_5'], color)
            ax.plot(grp['bin(risky/safe)'], grp['model_mean'], color=color, linewidth=1.5)
            # empirical
            ax.errorbar(grp['bin(risky/safe)'], grp['empirical_mean'],
                        yerr=grp['empirical_sem'], fmt='o', color=color,
                        markersize=3, capsize=2)
            ax.axhline(0.5, color='k', linestyle='--', linewidth=0.7, alpha=0.5)
            if ri == 0:
                ax.set_title(order)
            if ci == 0:
                ax.set_ylabel(f'{stake}\nProp. risky')
            finish(ax)
    axes[-1, 0].set_xlabel('log(risky/safe) bin')
    axes[-1, 1].set_xlabel('log(risky/safe) bin')
    fig.suptitle('Figure 2A — PMC model PPC', fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 2B ─────────────────────────────────────────────────────────────
    df = sheets['Figure 2B']
    fig, ax = plt.subplots(figsize=(5, 3))
    y = np.arange(len(df))
    ax.barh(y, df['posterior_mean'], xerr=df['posterior_sd'],
            color='steelblue', alpha=0.8, capsize=3, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df['parameter'], fontsize=7)
    ax.axvline(0, color='k', linewidth=0.8)
    finish(ax, xlabel='Posterior mean (combined sessions)', title='Figure 2B — Group-level parameters')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 2C ─────────────────────────────────────────────────────────────
    df = sheets['Figure 2C']
    comparisons = df['comparison'].unique()
    fig, axes = plt.subplots(1, len(comparisons), figsize=(10, 4), sharey=False)
    for ax, comp in zip(axes, comparisons):
        grp = df[df['comparison'] == comp].sort_values('posterior_mean')
        y = np.arange(len(grp))
        sig = grp['p_positive'] > 0.95
        colors = ['#E6531D' if s else '#3B82C4' for s in sig]
        ax.barh(y, grp['posterior_mean'], left=0,
                xerr=[grp['posterior_mean'] - grp['hdi_2_5'],
                      grp['hdi_97_5'] - grp['posterior_mean']],
                color=colors, alpha=0.8, capsize=2, height=0.7)
        ax.axvline(0, color='k', linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"sub-{s}" for s in grp['subject']], fontsize=6)
        finish(ax, title=grp['description'].values[0], xlabel='Posterior difference')
    fig.suptitle('Figure 2C — Subject-level parameter differences', fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 3A-D ───────────────────────────────────────────────────────────
    df = sheets['Figure 3A-D']
    models = df['model'].unique()
    fig, big_axes = plt.subplots(len(models), 2, figsize=(8, 10), sharey=True, sharex=True)
    if len(models) == 1:
        big_axes = big_axes[np.newaxis, :]
    for mi, model in enumerate(models):
        mdf = df[df['model'] == model]
        # Just plot "Small" stake for compactness
        stake = 'Small (5-7)'
        for ci, order in enumerate(orders):
            ax = big_axes[mi, ci]
            grp = mdf[(mdf['stake_bin'] == stake) & (mdf['order'] == order)].sort_values('bin(risky/safe)')
            color = ORDER_COLORS[order]
            errband(ax, grp['bin(risky/safe)'], grp['model_mean'],
                    grp['model_hdi_2_5'], grp['model_hdi_97_5'], color)
            ax.plot(grp['bin(risky/safe)'], grp['model_mean'], color=color, linewidth=1.5)
            ax.errorbar(grp['bin(risky/safe)'], grp['empirical_mean'],
                        yerr=grp['empirical_sem'], fmt='o', color=color,
                        markersize=3, capsize=2)
            ax.axhline(0.5, color='k', linestyle='--', linewidth=0.7, alpha=0.5)
            if mi == 0:
                ax.set_title(order)
            if ci == 0:
                ax.set_ylabel(f'{model[:20]}\nProp. risky', fontsize=7)
            finish(ax)
    big_axes[-1, 0].set_xlabel('log(risky/safe) bin')
    big_axes[-1, 1].set_xlabel('log(risky/safe) bin')
    fig.suptitle('Figure 3A-D — Alternative model PPCs (Small stake bin)', fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 3E ─────────────────────────────────────────────────────────────
    df = sheets['Figure 3E'].sort_values('elpd_loo', ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    y = np.arange(len(df))
    ax.barh(y, df['elpd_loo'] - df['elpd_loo'].max(),
            xerr=df['dse'], color='steelblue', alpha=0.8, capsize=3, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df['model'], fontsize=8)
    ax.axvline(0, color='k', linewidth=0.8)
    finish(ax, xlabel='ΔELPD-LOO (vs best model)', title='Figure 3E — Model comparison')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 4B ─────────────────────────────────────────────────────────────
    df = sheets['Figure 4B']
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    for ax, col, ylabel in zip(
        axes,
        ['r_decoded_vs_actual_numerosity', 'r_decoded_uncertainty_vs_abs_error'],
        ['r (decoded vs actual numerosity)', 'r (uncertainty vs |error|)']
    ):
        for scanner, grp in df.groupby('scanner'):
            ax.scatter(grp['subject'], grp[col],
                       color=SCANNER_COLORS[scanner], label=scanner, s=25, alpha=0.8)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.legend(frameon=False)
        finish(ax, xlabel='Subject', ylabel=ylabel)
    fig.suptitle('Figure 4B — Decoding accuracy', fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 4C ─────────────────────────────────────────────────────────────
    df = sheets['Figure 4C']
    fig, ax = plt.subplots(figsize=(3, 3))
    colors = {'High': '#E6531D', 'Low': '#3B82C4'}
    for i, (_, row) in enumerate(df.iterrows()):
        ax.bar(i, row['mean_distance_to_risk_neutral'],
               color=colors[row['neural_uncertainty']], alpha=0.8)
        ax.errorbar(i, row['mean_distance_to_risk_neutral'],
                    yerr=[[row['mean_distance_to_risk_neutral'] - row['hdi_2_5']],
                          [row['hdi_97_5'] - row['mean_distance_to_risk_neutral']]],
                    color='k', capsize=4, linewidth=1.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['neural_uncertainty'])
    finish(ax, xlabel='Neural uncertainty', ylabel='Distance to risk neutrality',
           title='Figure 4C')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 4D ─────────────────────────────────────────────────────────────
    df = sheets['Figure 4D']
    fig, ax = plt.subplots(figsize=(5, 3.5))
    y = np.arange(len(df))
    sig = df['p_significant (>0.95 or <0.05)']
    colors = ['#E6531D' if s else '#3B82C4' for s in sig]
    ax.barh(y, df['slope_mean'],
            xerr=[df['slope_mean'] - df['hdi_2_5'],
                  df['hdi_97_5'] - df['slope_mean']],
            color=colors, alpha=0.8, capsize=3, height=0.6)
    ax.axvline(0, color='k', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df['parameter'], fontsize=8)
    finish(ax, xlabel='Slope (effect of decoded neural uncertainty)',
           title='Figure 4D  (orange = p>0.95)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Figure 5B ─────────────────────────────────────────────────────────────
    df = sheets['Figure 5B']
    stake_bins_sym = sorted(df['stake_bin'].unique())
    orders_sym = sorted(df['order'].unique())
    fig, axes = plt.subplots(len(stake_bins_sym), len(orders_sym),
                             figsize=(7, 2.5 * len(stake_bins_sym)),
                             sharey=True, sharex=False)
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]
    order_colors_sym = {o: list(ORDER_COLORS.values())[i]
                        for i, o in enumerate(orders_sym)}
    for ri, stake in enumerate(stake_bins_sym):
        for ci, order in enumerate(orders_sym):
            ax = axes[ri, ci]
            grp = df[(df['stake_bin'] == stake) & (df['order'] == order)].sort_values('log_risky_safe_ratio_bin')
            color = order_colors_sym.get(order, 'steelblue')
            ax.errorbar(grp['log_risky_safe_ratio_bin'], grp['mean'], yerr=grp['sem'],
                        color=color, marker='o', markersize=3, linewidth=1.5, capsize=2)
            ax.axhline(0.5, color='k', linestyle='--', linewidth=0.7, alpha=0.5)
            if ri == 0:
                ax.set_title(order, fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{stake}\nProp. risky', fontsize=7)
            finish(ax)
    for ci in range(len(orders_sym)):
        axes[-1, ci].set_xlabel('log(risky/safe)')
    fig.suptitle('Figure 5B — Symbolic experiment', fontweight='bold')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved to {OUTPUT}")
