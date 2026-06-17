"""Figure 1D -- the theoretical PMCM schematic ("how Bayesian inference works").

This is a code-driven *illustration* (not data): scipy Gaussians for the
likelihood of each option, a shared prior, the Bayesian-updated posteriors, and
the expected-value comparison. Ported verbatim from
``risk_experiment/figures/likelihood_prior_revision.ipynb`` -- it draws no data
values, so nothing here is a reported number. Two scenarios (which option is
presented first carries the noisier evidence) plus the RNP-vs-safe-option
illustration.

Panels (per-panel vector PDF ingredients, assembled in Affinity):
  figure_01D_risky_first   -- risky option presented first (noisier)
  figure_01D_safe_first    -- safe option presented first (noisier)
  figure_01D_rnp           -- implied risk-neutral probability vs safe option

Usage:
    python -m risk_experiment.figures.figure_01D_model_schematic
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

from risk_experiment.figures import style

X = np.linspace(1.0, 9.0, 1000)


def get_posterior(mu_likelihood, sd_likelihood, mu_prior, sd_prior):
    """Conjugate Gaussian posterior (precision-weighted), as used by the model."""
    var_l, var_p = sd_likelihood ** 2, sd_prior ** 2
    mu = (mu_likelihood / var_l + mu_prior / var_p) / (1 / var_l + 1 / var_p)
    sd = np.sqrt(1.0 / (1 / var_l + 1 / var_p))
    return mu, sd


# Every distribution is drawn with the SAME integral (area). A Gaussian pdf
# already integrates to 1, so scaling them all by one common constant keeps the
# areas equal -- and then peak height scales as 1/sd automatically: a wider
# (noisier) representation is a LOW, broad bump and a narrower (more precise)
# one a TALL, thin bump, as it must be for probability densities. AREA_SCALE is
# tuned so the narrowest distribution in the schematic (sd ~ 0.15) peaks
# comfortably within one ROW_GAP without touching the row above.
AREA_SCALE = 0.52


def plot_dist(ax, mu, sd, ybase=0.0, xshift=0.0, color=None, shade=True, **kw):
    """Draw one distribution sitting on row baseline ``ybase`` (shifted right by
    ``xshift`` for the diagonal cascade). All distributions share the same
    integral (``AREA_SCALE``), so width and height trade off correctly. Returns
    the peak y, for label placing."""
    p = ss.norm(loc=mu, scale=sd).pdf(X) * AREA_SCALE
    ax.plot(xshift + X, ybase + p, color=color, alpha=0.8, lw=1.0, **kw)
    if shade:
        ax.fill_between(xshift + X, ybase, ybase + p, alpha=0.35, color=color)
    return ybase + p.max()


# Vertical space between the four stages (rows). Comfortably > the tallest
# (precise) distribution peak, so the bumps never reach the line above AND the
# 2nd-option label above that tall peak still clears the stage line overhead.
ROW_GAP = 2.2


def _inference_schematic(first='risky'):
    """The PMCM intuition cartoon for one presentation order (illustration)."""
    style.set_style()
    plt.rcParams['font.size'] = 6.0  # tiny labels so the drawing dominates
    palette = sns.color_palette('coolwarm', 4)[::-1]
    c_risky, c_safe = palette[0], palette[3]
    G = ROW_GAP

    fig, ax = plt.subplots(figsize=(48 * style.MM, 52 * style.MM),
                           constrained_layout=True)
    header = 'Risky option first' if first == 'risky' else 'Safe option first'
    ax.set_title(header, fontsize=8, fontweight='bold', fontstyle='italic', pad=4)

    # Fixed payoff positions; the FIRST-presented option carries the noisier
    # (wider) likelihood.
    mu_risky, mu_safe, mu_prior, sd_prior = 7.0, 3.0, 5.0, 0.75
    if first == 'risky':
        sd_risky, sd_safe = 0.75, 0.15
        ev_text = 'EV[safe] < EV[risky]'
    else:
        sd_risky, sd_safe = 0.15, 0.75
        ev_text = 'EV[risky] < EV[safe]'
    mu_pr, sd_pr = get_posterior(mu_risky, sd_risky, mu_prior, sd_prior)
    mu_ps, sd_ps = get_posterior(mu_safe, sd_safe, mu_prior, sd_prior)

    # First-presented (row 0) vs second-presented (row 1) option.
    if first == 'risky':
        f_mu, f_sd, f_c, f_lbl = mu_risky, sd_risky, c_risky, 'Risky payoff'
        s_mu, s_sd, s_c, s_lbl = mu_safe, sd_safe, c_safe, 'Safe payoff'
    else:
        f_mu, f_sd, f_c, f_lbl = mu_safe, sd_safe, c_safe, 'Safe payoff'
        s_mu, s_sd, s_c, s_lbl = mu_risky, sd_risky, c_risky, 'Risky payoff'

    y = [0.0, -G, -2 * G, -3 * G]   # stage baselines
    # No horizontal shift between stages: x is MAGNITUDE, shared across every
    # row, so the prior, both likelihoods and both posteriors line up on one
    # axis and the central-tendency pull toward the prior mean is legible.
    xs = [0, 0, 0, 0]

    for i in range(4):
        ax.plot([xs[i] + 1, xs[i] + 9], [y[i], y[i]], c='gray', lw=0.6, zorder=1)

    # Vertical reference at the prior mean -- the common target the noisy
    # representation is dragged toward. Spans the perceptual + inference stages.
    ax.plot([mu_prior, mu_prior], [y[2] - 0.35, G * 0.95], ls=':', c='0.55',
            lw=0.8, zorder=0)
    ax.annotate('Prior\nmean', (mu_prior, G * 0.95), ha='center', va='bottom',
                color='0.5', fontsize=5.5, linespacing=0.9)

    # Likelihoods (rows 0, 1); prior + posteriors (row 2).
    pk_f = plot_dist(ax, f_mu, f_sd, y[0], xs[0], color=f_c)
    pk_s = plot_dist(ax, s_mu, s_sd, y[1], xs[1], color=s_c)
    pk_prior = plot_dist(ax, mu_prior, sd_prior, y[2], xs[2], color='gray')
    plot_dist(ax, mu_pr, sd_pr, y[2], xs[2], color=c_risky)
    plot_dist(ax, mu_ps, sd_ps, y[2], xs[2], color=c_safe)

    # Dashed connectors: each likelihood informs the inference row.
    ax.plot([xs[0] + f_mu, xs[2] + f_mu], [y[0], y[2]], ls='--', c=f_c, lw=0.8,
            zorder=0)
    ax.plot([xs[1] + s_mu, xs[2] + s_mu], [y[1], y[2]], ls='--', c=s_c, lw=0.8,
            zorder=0)

    # Central-tendency MOVE on the inference row (the CRUX of the schematic),
    # drawn as a before -> after dumbbell so the shift itself is what reads, not
    # an arrowhead: a HOLLOW dot at the raw sensory reading (likelihood mean), a
    # FILLED dot at the Bayesian estimate (posterior mean), joined by a thick
    # black-outlined bar = the pull toward the prior. The noisy option's long bar
    # vs the precise option's near-zero move reads instantly and swaps between
    # the two panels.
    import matplotlib.patheffects as pe
    for mu_lik, mu_post, c in [(mu_risky, mu_pr, c_risky), (mu_safe, mu_ps, c_safe)]:
        x0, x1 = xs[2] + mu_lik, xs[2] + mu_post
        ax.plot([x0, x1], [y[2], y[2]], color=c, lw=3.0, solid_capstyle='round',
                zorder=7,
                path_effects=[pe.withStroke(linewidth=4.8, foreground='k')])
        ax.plot(x0, y[2], marker='o', mfc='white', mec='k', mew=1.2, ms=6.5,
                zorder=9)                                  # raw sensory reading
        ax.plot(x1, y[2], marker='o', mfc=c, mec='k', mew=1.2, ms=7.5,
                zorder=10)                                 # Bayesian estimate
        # Direction chevron on the bar -- only when the option actually moves, so
        # the noisy option shows a clear arrow toward the prior while the precise
        # option stays two near-coincident dots (the asymmetry is the point).
        d = x1 - x0
        if abs(d) > 0.4:
            ax.plot(x0 + 0.6 * d, y[2], marker=('>' if d > 0 else '<'),
                    color=c, mec='k', mew=0.8, ms=7, zorder=11)

    # Expected-value row: safe (p=1.0) and risky (p=0.55) projected down.
    ax.scatter([xs[3] + mu_ps], [y[3]], color=c_safe, zorder=10)
    ax.scatter([xs[3] + mu_pr * 0.55], [y[3]], color=c_risky, zorder=10)
    ax.plot([xs[2] + mu_ps, xs[3] + mu_ps], [y[2], y[3]], color=c_safe, lw=0.8)
    ax.plot([xs[2] + mu_pr, xs[3] + mu_pr * 0.55], [y[2], y[3]], color=c_risky,
            lw=0.8)

    # --- Stage labels (left of each row) ---
    ax.annotate('1st option', (xs[0] + 0.6, y[0]), ha='right',
                va='center')
    ax.annotate('2nd option', (xs[1] + 0.6, y[1]), ha='right',
                va='center')
    ax.annotate('Bayesian inference', (xs[2] + 0.6, y[2]), ha='right', va='center')
    ax.annotate('Expected value', (xs[3] + 0.6, y[3]), ha='right',
                va='center')

    # --- Floating labels, each above its own distribution peak (clear gaps) ---
    ax.annotate(f_lbl, (xs[0] + f_mu, pk_f + 0.18), ha='center', va='bottom')
    # 2nd-option label above its (tall, narrow) peak. With the shared magnitude
    # axis the peak sits over empty space in the row above, so there's room.
    ax.annotate(s_lbl, (xs[1] + s_mu, pk_s + 0.18), ha='center', va='bottom')
    ax.annotate('Prior', (xs[2] + mu_prior, pk_prior + 0.18),
                ha='center', va='bottom', color='0.45')
    ax.annotate('p=1.0', (xs[2] + mu_ps - 0.3, y[2] - 0.55 * G), ha='right',
                va='center', color=c_safe)
    ax.annotate('p=0.55', (xs[2] + mu_pr + 0.3, y[2] - 0.55 * G), ha='left',
                va='center', color=c_risky)
    ax.annotate(ev_text, (7.0, y[3] - 0.7), ha='center', va='top')

    ax.set_xlim(-3.6, 10.8)
    ax.set_ylim(y[3] - 1.3, pk_f + 0.8)
    ax.axis('off')
    name = f'figure_01D_{first}_first'
    # Tight crop with no padding: removes all surrounding margin so the panel
    # is just the schematic (<= ~48 mm wide, fits beside 1C within 180 mm).
    pdf = style.save_panel(fig, name, tight=True, pad=0.0)
    print(f'Wrote figure: {pdf}')


def _rnp_illustration():
    """Reproduce illustration_safe_vs_rnp.pdf (implied RNP vs safe option)."""
    style.set_style()
    ev = np.linspace(1, 9, 100)
    risky_options = ev / 0.55
    safe_options = ev
    mu_prior, sd_prior = (7.0 + 3.0) / 2, 0.75
    std_first, std_second = 0.75, 0.15

    def post_mean(vals, sd):
        return np.array([get_posterior(v, sd, mu_prior, sd_prior)[0] for v in vals])

    e_risky_first = post_mean(risky_options, std_first) * 0.55
    e_risky_second = post_mean(risky_options, std_second) * 0.55
    e_safe_first = post_mean(safe_options, std_first)
    e_safe_second = post_mean(safe_options, std_second)

    fig, ax = plt.subplots(figsize=(2.4, 2.8), constrained_layout=True)
    ax.plot(safe_options, e_risky_first / e_safe_second * 0.55, lw=1.5,
            label='Safe option first')
    ax.plot(safe_options, e_risky_second / e_safe_first * 0.55, lw=1.5,
            label='Risky option first')
    ax.axhline(0.55, c='k', ls='--', lw=0.8, label='Risk neutral')
    ax.set_ylabel('Risk-neutral probability')
    ax.set_xlabel('Safe option')
    ax.annotate('Risk-seeking', (9.2, 0.565), ha='right', va='bottom')
    ax.annotate('Risk-averse', (9.2, 0.525), ha='right', va='top')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    sns.despine(ax=ax, offset=3, trim=False)
    pdf = style.save_panel(fig, 'figure_01D_rnp')
    print(f'Wrote figure: {pdf}')


def main():
    _inference_schematic('risky')
    _inference_schematic('safe')
    _rnp_illustration()


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    main()
