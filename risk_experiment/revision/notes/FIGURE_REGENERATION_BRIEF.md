# Figure regeneration brief — NCOMMS-24-63995B

**Paper:** "Rapid Changes in Risk Attitudes Originate from Bayesian Inference on Parietal Magnitude Representations" (de Hollander et al.), accepted in principle at *Nature Communications*, final revision stage.

**Your job:** regenerate every main figure as a separate, submission-ready, vector file that satisfies the Nature Communications artwork rules and the editor's specific request that **no text be smaller than 6 pt at final printed size**. Work from the existing plotting code in the repo — modify it, don't redraw from memory. Where a panel is a hand-drawn schematic (not generated from data), flag it; do not fabricate it.

This is a formatting + layout pass, not a reanalysis. The numbers and statistics must not change. If you cannot reproduce a panel from the existing code/data, stop and report it rather than inventing data.

---

## 0. The one change to the figure set: split old Figure 3

The editor asked us to split panel **3E** (the ELPD model-comparison forest plot) out of Figure 3, because at full width it compresses panels 3A–D. We are doing this. The result is **6 main figures**, renumbered as follows:

| New | Was | Content |
|-----|-----|---------|
| Figure 1 | Figure 1 | Task paradigm + behavioural order×stake effects (panels A–D) |
| Figure 2 | Figure 2 | PMCM posterior predictions + group/participant parameter posteriors (A–C) |
| Figure 3 | Figure 3 A–D | Posterior-predictive plots for the four alternative models (A–D only) |
| **Figure 4** | **Figure 3 E** | **ELPD model comparison (the split-out panel, now standalone)** |
| Figure 5 | Figure 4 | nPRF mapping + decoding + neural-uncertainty→behaviour (A–D) |
| Figure 6 | Figure 5 | Symbolic (Arab-numeral) experiment paradigm + psychophysics (A–B) |

You produce six files: `Figure_1.pdf` … `Figure_6.pdf`. Do **not** renumber anything in the manuscript text — that is handled separately. Just produce correctly numbered files.

---

## 1. Locate the source first (do this before plotting)

1. Identify the repository and the scripts that currently generate each figure. The plotting code lives in the project's analysis repo (`ruffgroup/risk_experiment`); models are in `bauer`; decoding/nPRF in `braincoder`. Find the script or notebook responsible for each figure and read it before changing anything.
2. Identify the data each script consumes (derivatives on disk, behavioural data, posterior sample files). Confirm you can run each figure script end-to-end and reproduce the *current* version before reformatting.
3. Build an inventory: for each of the 6 figures, note (a) the script/function, (b) which panels are **data-driven** (regenerable) vs **schematic** (hand-made), (c) the data inputs. Report this inventory back before mass-editing.

**Schematic panels — do not regenerate from data:** `Figure 1A` (trial sequence), `Figure 1D` (PMCM intuition cartoon), and `Figure 6A` (symbolic trial sequence) are illustrator-style schematics, not plots. For these, only verify/repair formatting (vector, editable text ≥6 pt, RGB) on the existing source files. If their source isn't in the repo, flag them as needing manual handling and leave them out of the automated pipeline. (These three are also the panels the editor flagged as "suspected third-party content"; they must be confirmed author-created, so treat them carefully.)

---

## 2. Hard requirements (Nature Communications + editor)

Non-negotiable, every figure:

- **Minimum font size 6 pt at final size.** This is the editor's explicit instruction. Target ≥7 pt for body/tick text so there is no rounding risk; never let anything render below 6 pt. This includes tick labels, axis labels, in-panel annotations, colorbar labels, keys, and any text inside schematics.
- **Vector format, fonts embedded as editable text** — submit PDF. Set `pdf.fonttype = 42` (and `ps.fonttype = 42`, `svg.fonttype = 'none'`) so text stays selectable/editable in Illustrator, not converted to paths. Verify after export (see §6).
- **One file per figure**, all panels for that figure in the one file, panels labelled, fitting entirely on a single page.
- **Each figure fits within an A4 page** (210 × 297 mm) at final scale.
- **Sans-serif throughout** — Helvetica (house font; Arial only as last-resort fallback, and flag the substitution).
- **RGB colour mode** (online-first journal).
- **No figure legends/captions inside the figure file** — captions live in the manuscript. The figure may contain short panel titles and in-panel condition keys, but not the prose legend.
- **Define conditions with an in-panel key, not coloured symbols in the caption** (e.g., the "Safe first / Risky first" key belongs inside the panel).
- **Line weights ≥ 0.5 pt** so nothing drops out in print; data lines ~1–1.2 pt.

**Physical width** — design at the final width, never rescale the exported PDF afterwards (that rescales fonts and breaks the 6 pt floor):
- Single column ≈ 88 mm (3.46")
- 1.5 column ≈ 120 mm (4.72")
- Double column / full width ≈ 180 mm (7.09")

Pick the smallest width that keeps the panels legible; multi-panel figures here will mostly be double-column (180 mm). Keep height ≤ ~230 mm so the figure plus its (separately placed) legend fits A4.

**Raster sub-elements** (the brain-surface renders in Figure 5A): keep these at ≥ 300 dpi at final size, embedded — but keep all *text* around them (labels, colorbar ticks) as vector. A mixed vector+raster PDF is fine; the rule is only that text is vector.

---

## 3. House style (use the lab's existing aesthetic)

The figures follow the NYU vision-science house style (despined offset spines, outward ticks, direct labelling over legends, muted palette, posterior HDIs as shaded bands). Reuse it. Drop this rcParams block at the top of each figure script and adjust nothing downward below the 6 pt floor:

```python
import matplotlib as mpl
import seaborn as sns

mpl.rcParams.update({
    # Typography — Helvetica house font; sizes chosen to stay >= 7 pt at final scale
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8,            # never below 7; 6 is the absolute floor
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
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

    # Output: editable text in vector formats
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',

    # Figure
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
sns.set_context('paper')
```

Set figure size in inches at creation, e.g. `fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.5), constrained_layout=True)` for a double-column figure, and finish data axes with `sns.despine(offset=4, trim=True)` (for FacetGrid, pass `despine=False` then call `sns.despine` on the figure).

**Panel letters:** A, B, C… bold, ~8–9 pt, top-left **outside** the axes, same position on every panel:
```python
ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')
```

**Uncertainty:** these are Bayesian results — shaded **HDI/credible-interval** bands (not "confidence interval"), posterior mean/median as the central line; behavioural SEM bands across participants where applicable. Keep whatever the current scripts compute; just render it in this style.

**Palette:** keep the existing condition colours if they already encode meaning (Safe first vs Risky first; risk-seeking/neutral/averse). If they're a default `tab10`/`Set1`, replace with the muted hand-picked palette (`['#3B5BA5', '#C44E52', '#5D8C3F', '#8172B2', '#9C9C9C']`) and verify grayscale-distinguishability.

---

## 4. Per-figure notes

For each, regenerate the data panels and lay them out to the target width; verify the 6 pt floor.

- **Figure 1 (double col).** A: trial-sequence schematic (manual). B: prop. risky choices vs safe-offer magnitude, hue = order; SEM error bars across participants. C: psychophysical curves per stake size with RNP insets. D: PMCM intuition schematic (manual). Keep the Safe-first/Risky-first key in-panel.
- **Figure 2 (double col).** A: PMCM posterior-predictive curves, 3 stake sizes × order (markers = data, shaded = 95% CI). B: group-level posteriors (Evidence sd, Prior mu, Prior std), hue = option/risky-safe. C: participant-level posterior differences, coloured by risk category. Provide the colour key in-panel.
- **Figure 3 (double col).** A–D only: posterior-predictive plots for the four alternative models. **Remove old panel E** from this figure's layout and rebalance so A–D fill the space legibly.
- **Figure 4 (single or 1.5 col).** The standalone ELPD model-comparison forest plot (former 3E): models on y, ELPD on x, error bars = SE on ELPD. This is the panel we are giving room to breathe — make it comfortably legible.
- **Figure 5 (double col).** A: nPRF preferred-numerosity surface maps, 3T & 7T (raster ≥300 dpi; vector colorbar + labels). B: decoding scatters (corr of posterior mean vs numerosity; corr of posterior sd vs decoding error). C: distance-from-risk-neutral posteriors, low vs high neural uncertainty. D: posterior group-parameter estimates of the neural regressors (Evidence sd, Prior mean, Prior std).
- **Figure 6 (double col).** A: symbolic (Arab-numeral) trial-sequence schematic (manual). B: psychophysical curves per stake size with RNP and RNP-difference, hue = order.

> Note on legends: the editor also requested several caption-level additions (defining *n*, error-bar type, the statistical test, and exact *P* values for specific panels). Those are **manuscript caption edits, not figure-file changes** — you don't need to put them in the figures. But while you're in each script, **also export the underlying numbers** behind every data panel (one tidy table per panel) so they can be assembled into the Source Data Excel the journal requires.

---

## 5. Workflow

1. Reproduce each figure as-is from the existing code (sanity check).
2. Apply the rcParams block + target `figsize` + `sns.despine(offset=4, trim=True)` + panel letters.
3. For Figure 3/4: split the layout so 3E becomes the standalone Figure 4.
4. Replace any default palette; move legends in-panel; convert any caption text out of the figure.
5. Export each as `Figure_N.pdf` (vector, RGB, fonts type 42).
6. Run the verification checks (§6). Fix and re-export until all pass.
7. Export the per-panel source-data tables alongside.

---

## 6. Verification (run these automatically before declaring done)

For every exported PDF:

- **Minimum font size ≥ 6 pt.** Extract text font sizes and assert the minimum. One way:
  ```bash
  # list text + sizes; inspect the smallest
  pdffonts Figure_1.pdf            # confirm fonts are embedded (emb=yes) and are Helvetica/Arial, Type 42/TrueType, not "Type 3" (paths)
  mutool info Figure_1.pdf
  ```
  Programmatic size check with pikepdf/pdfminer is preferable — parse text objects and assert no rendered size < 6 pt at the page's true dimensions. If you can't measure reliably, render each PDF to PNG at 300 dpi and visually confirm the smallest label is legible, and report the measured figsize + font rcParams as evidence.
- **Fonts embedded as text, not paths.** `pdffonts` must not report `Type 3` for the body text; text must be selectable.
- **Page size fits A4** and matches the intended width (88/120/180 mm). Assert page width ≤ 180 mm (+ small margin) and height ≤ 230 mm.
- **RGB**, not CMYK.
- **Vector** for all line/text art (raster only inside the brain-map panel).
- **Grayscale check** (`convert -colorspace Gray`): conditions remain distinguishable.

Report a short table: figure → width(mm) × height(mm), min font size (pt), fonts embedded (y/n), vector (y/n). Flag anything failing.

---

## 7. Deliverables

- `Figure_1.pdf` … `Figure_6.pdf` — submission-ready vector figures.
- The (lightly) modified plotting scripts, committed.
- One source-data table per data panel (CSV or sheet) for assembling the Source Data workbook.
- The verification table from §6.
- A list of any panels you could **not** regenerate (expected: the three schematics 1A, 1D, 6A) with a note on what's needed.

## 8. Do not

- Do not change any analysis, statistic, parameter estimate, or data value.
- Do not rescale exported PDFs after the fact (`\includegraphics[width=...]` or Illustrator resize) — regenerate at the target width instead.
- Do not bake the prose legend into the figure.
- Do not invent or approximate a schematic panel you can't find the source for — flag it.
- Do not use default `tab10`/`Set1` palettes or capped error bars at print size.
