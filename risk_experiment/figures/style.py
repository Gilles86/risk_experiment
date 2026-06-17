"""House style for the manuscript figures (NCOMMS-24-63995B revision).

Single source of truth for the publication aesthetic so every
``figure_NN_*.py`` script produces consistent, submission-ready vector
*ingredients* (one PDF per data panel). Final multi-panel composites are
assembled in Affinity Designer; panel letters and the hand-drawn
schematics (Fig 1A/1D, 6A) and brain renders (5A) are added there.

Hard requirements encoded here (Nature Communications + editor):
  * Minimum font size 6 pt at final size -- we design at >= 7 pt so there
    is no rounding risk.
  * Vector PDF with fonts embedded as editable text (Type 42), never paths.
  * Sans-serif throughout (Helvetica house font).
  * RGB colour, despined/offset axes, outward ticks, line weights >= 0.5 pt.

Nothing here changes any number or statistic -- it is form only.
"""
import os.path as op

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# Muted, colourblind-safe, grayscale-distinguishable categorical palette.
# Keep the existing semantic condition colours where they already encode
# meaning (Safe first / Risky first; risk-seeking / neutral / averse).
PALETTE = ['#3B5BA5', '#C44E52', '#5D8C3F', '#8172B2', '#9C9C9C']

# Existing condition colours used across the paper -- do not recolour these,
# they carry meaning the reader has already learned.
ORDER_COLORS = {'Safe first': '#1f77b4', 'Risky first': '#ff7f0e'}
RISKTYPE_COLORS = {'Risk-seeking': '#C44E52',
                   'Risk-averse': '#4C9A8E',
                   'Risk-neutral': '#000000'}

# Physical widths (inches) -- pick at figure creation, never rescale on export.
WIDTH_SINGLE = 3.46   # 88 mm  single column
WIDTH_ONEHALF = 4.72  # 120 mm 1.5 column
WIDTH_DOUBLE = 7.09   # 180 mm double column


RC = {
    # Typography -- Helvetica house font. Sizes raised so the smallest text is
    # 8 pt at final scale: comfortably above the 6 pt floor with headroom for
    # any slight down-scaling during Affinity assembly.
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'mathtext.fontset': 'stixsans',

    # Axes
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelpad': 3,

    # Ticks: outward, short, thin
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,

    # Lines and markers
    'lines.linewidth': 1.0,
    'lines.markersize': 3.5,
    'patch.linewidth': 0.5,

    # Legend
    'legend.frameon': False, 'legend.handlelength': 1.5,

    # Output: editable text in vector formats (Type 42, not paths)
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',

    # Figure
    'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
}


def shim_pymc_for_bauer(pm):
    """Let analysis-era bauer 0.1.0 (pinned at src/bauer @ e246d78) build models
    under a newer PyMC than the 5.10.3 it was written against.

    The only break is ``pm.Data(..., mutable=True)``: the ``mutable`` kwarg was
    removed in PyMC >= 5.16 (data containers are always mutable now). Dropping it
    is pure plumbing -- it does not touch the likelihood, so ``compute_log_
    likelihood`` / PPC values are identical to the analysis env. No-op when the
    running PyMC still accepts ``mutable``. Call once before building any bauer
    model, e.g.::

        import pymc as pm
        from risk_experiment.figures import style
        style.shim_pymc_for_bauer(pm)
    """
    import inspect
    try:
        if 'mutable' in inspect.signature(pm.Data).parameters:
            return
    except (ValueError, TypeError):
        pass
    _orig_data = pm.Data

    def _data(*args, **kwargs):
        kwargs.pop('mutable', None)
        return _orig_data(*args, **kwargs)

    pm.Data = _data


def set_style():
    """Apply the house rcParams. Call once at the top of each figure script."""
    _register_helvetica_bold()
    mpl.rcParams.update(RC)
    sns.set_context('paper')
    sns.set_style('ticks')
    mpl.rcParams.update(RC)  # re-apply: sns.set_* overrides some keys
    return PALETTE


# Output lives IN the repo (version-controlled submission deliverables),
# separate from the input data under bids_folder. Panels + per-panel source
# data are assembled into the final multi-panel figures in Affinity Designer.
REPO_ROOT = op.abspath(op.join(op.dirname(__file__), op.pardir, op.pardir))
FIGURES_OUT = op.join(REPO_ROOT, 'risk_experiment', 'revision', 'figures')


# Fixed geometry shared by the Figure 2 ingredient rows (2A / 2B / 2C). Using
# the SAME figsize and the SAME margins (instead of constrained_layout, which
# optimises each figure separately) makes the three saved PDFs byte-identical in
# panel position and size -- so they stack in Affinity with aligned columns.
# Save these with save_panel(..., tight=False) so the fixed margins are kept.
FIG2_MARGINS = dict(left=0.09, right=0.985, wspace=0.26, top=0.82, bottom=0.22)


def figure2_row(ncols=3, height_in=2.1, sharey=False):
    """A 1xN double-column row with the shared fixed Figure-2 geometry."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, ncols, figsize=(WIDTH_DOUBLE, height_in),
                             sharey=sharey)
    fig.subplots_adjust(**FIG2_MARGINS)
    return fig, axes


_RATIO_BIN_MEANS = None


def ratio_bin_means(bids_folder='/data/ds-risk', n_bins=7):
    """Mean risky/safe ratio for each per-subject qcut bin index (0..n_bins-1).

    The PPC panels (Fig 2A, Fig 3) plot proportion-risky against the *bin index*
    of the per-subject 7-quantile split of ``n_risky/n_safe`` (the exact binning
    in ``figure_03_alt_models.compute_summary``). This returns the mean natural-
    space ratio per bin index, so a bin position can be relabeled with the real
    ratio it represents -- form only, it never moves a plotted point. Memoised
    (the mapping is fixed across the cohort).
    """
    global _RATIO_BIN_MEANS
    if _RATIO_BIN_MEANS is not None:
        return _RATIO_BIN_MEANS
    import numpy as np
    import pandas as pd
    from risk_experiment.utils.data import get_all_behavior
    df = get_all_behavior(bids_folder=bids_folder)
    ratio = df['n_risky'] / df['n_safe']
    binned = (df.assign(_r=ratio)
              .groupby('subject', group_keys=False)
              .apply(lambda x: pd.qcut(x['_r'], q=n_bins, labels=False,
                                       duplicates='drop')))
    _RATIO_BIN_MEANS = ratio.groupby(binned).mean()
    return _RATIO_BIN_MEANS


def ratio_bin_ticks(bids_folder='/data/ds-risk', nice=(1.5, 2, 2.5, 3), n_bins=7):
    """Tick (positions, labels) placing nicely-rounded natural-space risky/safe
    ratios on the binned (0..n_bins-1) PPC x-axis (Fig 2A, Fig 3).

    Positions come from log-interpolating each nice ratio against the per-bin
    mean ratios, so e.g. ``2`` lands exactly where the bins say ratio 2 sits.
    """
    import numpy as np
    means = ratio_bin_means(bids_folder, n_bins)
    bins = means.index.values.astype(float)
    logm = np.log(means.values)
    lo, hi = means.values.min(), means.values.max()
    keep = [r for r in nice if lo <= r <= hi]
    pos = np.interp(np.log(keep), logm, bins)
    return pos, [f'{r:g}' for r in keep]


def ratio_pos(ratio, bids_folder='/data/ds-risk', n_bins=7):
    """Fractional position of a natural-space risky/safe ratio on the binned
    (0..n_bins-1) PPC x-axis -- for reference lines, e.g. the risk-neutral ratio
    ``1/0.55`` where the risky option's expected value equals the safe offer.
    """
    import numpy as np
    means = ratio_bin_means(bids_folder, n_bins)
    return float(np.interp(np.log(ratio), np.log(means.values),
                           means.index.values.astype(float)))


def figures_dir(bids_folder=None):
    """Repo dir holding the vector panel ingredients (PDF + SVG)."""
    return FIGURES_OUT


def source_data_dir(bids_folder=None):
    """Per-panel source-data tables (for the journal's Source Data workbook)."""
    return op.join(FIGURES_OUT, 'source_data')


MM = 1 / 25.4  # millimetres -> inches, for exact panel sizing

# Convenience for genuinely bold text. macOS ships Helvetica as a .ttc whose
# bold face matplotlib won't index on its own (so fontweight='bold' silently
# renders regular); _register_helvetica_bold() fixes that, so plain
# fontweight='bold' then yields true Helvetica Bold.
BOLD = dict(fontweight='bold')


def _register_helvetica_bold():
    """Extract the Bold face from macOS Helvetica.ttc and register it with
    matplotlib so fontweight='bold' renders real Helvetica Bold (matplotlib
    indexes only face 0 of a .ttc otherwise). No-op if unavailable or already
    done."""
    import os
    from matplotlib import font_manager as fm
    from matplotlib.font_manager import findfont, FontProperties

    # Already distinct? nothing to do.
    try:
        reg = findfont(FontProperties(family='Helvetica', weight='normal'))
        bold = findfont(FontProperties(family='Helvetica', weight='bold'))
        if reg != bold:
            return
    except Exception:
        return

    cache = op.expanduser('~/.cache/risk_fonts/Helvetica-Bold.ttf')
    try:
        if not op.exists(cache):
            from fontTools.ttLib import TTCollection
            src = '/System/Library/Fonts/Helvetica.ttc'
            if not op.exists(src):
                return
            os.makedirs(op.dirname(cache), exist_ok=True)
            TTCollection(src).fonts[1].save(cache)  # face 1 = Helvetica Bold
        fm.fontManager.addfont(cache)
    except Exception:
        pass  # bold falls back to regular Helvetica; not fatal


def save_panel(fig, name, bids_folder=None, svg=True, tight=True,
               transparent=False, pad=0.02):
    """Save a panel ingredient as vector PDF (+ SVG) into the repo figures dir.

    Returns the PDF path. Text stays editable (Type 42); raster only where a
    panel embeds a bitmap (e.g. brain renders), never for line/text art.

    ``tight=True`` crops to the content bounding box (default). Set
    ``tight=False`` to emit a PDF whose page size equals the figure ``figsize``
    exactly -- use this when a panel must hit a precise mm box for Affinity
    assembly (rely on ``constrained_layout`` to pack labels inside the box).
    """
    import os
    import matplotlib as mpl
    out_dir = figures_dir()
    os.makedirs(out_dir, exist_ok=True)
    pdf = op.join(out_dir, f'{name}.pdf')
    svgp = op.join(out_dir, f'{name}.svg')
    # transparent=True drops the white figure/axes background so the ingredient
    # can be overlaid in Affinity without masking what's behind it.
    if tight:
        fig.savefig(pdf, bbox_inches='tight', pad_inches=pad,
                    transparent=transparent)
        if svg:
            fig.savefig(svgp, bbox_inches='tight', pad_inches=pad,
                        transparent=transparent)
    else:
        # Exact: page size == figsize. Force the 'standard' (no-crop) bbox via
        # rc_context, since the global rcParam is 'tight' and a None arg would
        # fall back to it ('standard' is not a valid savefig() argument value).
        with mpl.rc_context({'savefig.bbox': 'standard'}):
            fig.savefig(pdf, transparent=transparent)
            if svg:
                fig.savefig(svgp, transparent=transparent)
    return pdf
