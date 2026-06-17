# REPRODUCE.md — reproduction guide for de Hollander et al. (Nature Communications)

**"Rapid Changes in Risk Attitudes Originate from Bayesian Inference on Parietal
Magnitude Representations"** (de Hollander, G., et al.).

This is the step-by-step guide for a human to reproduce the analyses and the six
main figures of the published paper. It complements the conceptual `README.md`
(which explains the models): this file is the *operational* recipe — what to
install, in what order to run things, and how to regenerate and verify each
figure.

> **Figure numbering.** This guide uses the **final (published) figure
> numbering** (six main figures). The previous-revision numbering and the
> notebooks under `README.md` use an older scheme — see the
> [old→new mapping](#oldnew-figure-mapping) below.

---

## 1. Environment

### Conda environment `risk7t`

All analysis runs in the conda env defined by [`environment.yml`](environment.yml):

```bash
conda env create -f environment.yml      # creates env "risk7t" (python 3.10)
conda activate risk7t
pip install -e .                          # installs the risk_experiment package
```

The interpreter used throughout this guide:

```
~/mambaforge/envs/risk7t/bin/python
```

Core stack: TensorFlow 2.14 (+ tensorflow-probability) for `braincoder`
encoding/decoding, PyMC / ArviZ / Bambi for the Bayesian behavioural models,
nilearn / nibabel for fMRI I/O, seaborn / matplotlib for figures.

### The two in-house libraries

| Library | Role | How it is pinned |
|---|---|---|
| [`braincoder`](https://github.com/Gilles86/braincoder) | nPRF encoding models + Bayesian decoding (Fig 5) | pip-installed from GitHub `@main` (see `environment.yml`) |
| [`bauer`](https://github.com/ruffgroup/bauer) | Bayesian behavioural models (PMCM + alternatives, Figs 2–4) | **analysis-era 0.1.0 pinned as a git worktree** |

#### bauer 0.1.0 git worktree at `src/bauer`

The behavioural models were fit and the figure scripts rebuild their model
objects against **bauer 0.1.0**, pinned to commit **`e246d78`**. To reproduce
the figures faithfully, check out exactly that commit as a worktree at
`src/bauer` and install it editable:

```bash
# run from the risk_experiment repo root; <bauer-clone> = your local bauer checkout
git clone https://github.com/ruffgroup/bauer.git ~/git/bauer        # if you don't have it
git -C ~/git/bauer worktree add "$(pwd)/src/bauer" e246d78
pip install -e "$(pwd)/src/bauer"                                    # inside the risk7t env
```

`src/` is gitignored (it is an editable source checkout, not repo content), so it
does not show up in `git status`. The figure scripts import `bauer` indirectly
via `risk_experiment.cogmodels.fit_model` / `fit_probit`.

#### The PyMC compatibility shim

bauer 0.1.0 was written against PyMC 5.10.3. Under newer PyMC the only break is
`pm.Data(..., mutable=True)` (the `mutable` kwarg was removed in PyMC ≥ 5.16).
`risk_experiment/figures/style.py` provides
[`shim_pymc_for_bauer(pm)`](risk_experiment/figures/style.py) which drops that
kwarg — pure plumbing that does **not** touch the likelihood, so
`compute_log_likelihood` / PPC values are identical to the analysis env. Every
figure script that builds a bauer model calls it:

```python
import pymc as pm
from risk_experiment.figures import style
style.shim_pymc_for_bauer(pm)          # no-op if PyMC still accepts mutable=
```

If you reproduce inside the *exact* analysis env (PyMC 5.10.3), the shim is a
no-op and can be ignored.

---

## 2. Data

### 2.0 Where to get it (cold start)

The data are public. Download them and point `--bids_folder` at the result
(everything below assumes `/data/ds-risk` and `/data/ds-symbolicrisk`):

| Dataset | Source | Contents |
|---|---|---|
| Main 7T/3T experiment (`ds-risk`) | **OpenNeuro `ds007508`** — https://openneuro.org/datasets/ds007508 (DOI 10.18112/openneuro.ds007508.v1.0.0) | Raw BIDS + `derivatives/` (fMRIPrep, fitted `cogmodels/` traces, `encoding_model*`, `decoded_pdfs*`). |
| Symbolic (Arabic-numeral) experiment (`ds-symbolicrisk`) | **figshare** — https://doi.org/10.6084/m9.figshare.31400430 | Behavioural data + fitted `derivatives/risk_model/psychophysical/model*_samples.nc`. |

```bash
# example (OpenNeuro CLI or datalad); then:
ln -s /path/to/ds007508        /data/ds-risk
ln -s /path/to/ds-symbolicrisk /data/ds-symbolicrisk
```

If the published `derivatives/` are included in the download, you can reproduce
**all figures and statistics without any refitting** (skip §3 stages 1–5). The
fitted-trace table in §2 below lists exactly which files the figures/stats load.

### 2.1 Layout

The analyses read a BIDS dataset plus its derivatives:

```
/data/ds-risk/                     # main (non-symbolic) experiment, N=30
├── sub-{id}/ses-{3t1,3t2,7t1,7t2}/{func,anat,beh}/
└── derivatives/
    ├── fmriprep/                  # fMRIPrep preprocessing
    ├── cogmodels/                 # ★ fitted behavioural model traces:
    │                              #   model-{label}_trace.netcdf
    ├── encoding_model*.natural_space/        # per-voxel nPRF parameters
    └── decoded_pdfs*.natural_space/          # trial-by-trial decoded numerosity (E, sd)

/data/ds-symbolicrisk/             # symbolic (Arabic-numeral) experiment (Fig 6)
└── derivatives/risk_model/psychophysical/model{N}_samples.nc
```

The **single source of truth for all data access** is the `Subject` class in
[`risk_experiment/utils/data.py`](risk_experiment/utils/data.py)
(plus module-level helpers `get_all_behavior`, `get_all_subjects`,
`get_all_subject_ids`). All scripts take `--bids-folder` (default `/data/ds-risk`).

> **Note.** sub-24 is always excluded (claustrophobia — only ever completed one
> session). The analysed sample is sub-02…sub-32 minus sub-24.

The fitted **model traces** (`derivatives/cogmodels/model-*_trace.netcdf`) are the
expensive intermediate. Figures load these directly; you do **not** need to refit
the models to regenerate figures. The trace labels you need:

| Trace | Model | Used by |
|---|---|---|
| `model-12_trace.netcdf` | **PMCM** main model (6 params × session) | Fig 2, Fig 4 |
| `model-klw_trace.netcdf` | Alt A: shared prior, equal noise | Fig 3, Fig 4 |
| `model-52_trace.netcdf` | Alt B: varying priors, equal noise | Fig 3, Fig 4 |
| `model-42_trace.netcdf` | Alt C: shared prior, varying noise | Fig 3, Fig 4 |
| `model-eu_trace.netcdf` | Alt D: expected-utility model | Fig 3, Fig 4 |
| `model-neural32_trace.netcdf` | PMCM + decoded `sd` × session regressors | Fig 5 |
| `model-probit_full_trace.netcdf` | Psychophysical probit (RNP insets) | Fig 1 |

---

## 3. Pipeline stages (in order)

Run these in order to rebuild the dataset from scratch. **To reproduce only the
figures, skip to §4** — stages 1–4 produce the derivatives that the published
traces already capture.

### Stage 1 — raw → BIDS preparation

`risk_experiment/prepare/` and the revision BIDS fixer:

- `prepare/cleanup_raw_data.py` (+ `batch_cleanup_raw_data.py`) — DICOM/raw → BIDS.
- `prepare/prepare_behavior.py` — parse behavioural logs → per-run `events.tsv`,
  sync to fMRI runs.
- `prepare/correct_intendedfor.py`, `prepare/correct_trs.py`,
  `prepare/check_tsnr.py`, `prepare/make_subject_images.py` — metadata/QC fixes.
- `risk_experiment/revision/fix_all_bids.py` — the consolidated BIDS-validation
  fixer (IntendedFor paths, events durations, JSON metadata, NIfTI temporal
  units, RepetitionTime sync). See `risk_experiment/revision/README_CLEANUP.md`.

### Stage 2 — preprocessing (fMRIPrep) + surface

- `risk_experiment/cluster_preprocess/` — fMRIPrep batch/cluster wrappers.
- `risk_experiment/preproc/smooth_surf.py` — surface smoothing.
- `risk_experiment/registration/`, `risk_experiment/surface/` — anatomical
  registration, IPS mask creation, pRF centre-of-mass.

### Stage 3 — GLMs (single-trial betas)

`risk_experiment/glms/`:

- `fit_single_trials_volume.py` / `..._volume_denoise.py` / `..._surf.py` —
  single-trial GLM betas (GLMsingle/denoise variants).
- `simple_mapper.py`, `simple_task.py` — mapper / task GLMs.

### Stage 4 — encoding / decoding (nPRF; Fig 5)

`risk_experiment/encoding_model/` (uses `braincoder`):

```bash
python -m risk_experiment.encoding_model.fit_mapper  <subject> <session>   # nPRF on localiser
python -m risk_experiment.encoding_model.fit_task    <subject> <session>   # nPRF on task
python -m risk_experiment.encoding_model.decode      <subject> <session>   # invert → per-trial E, sd
```

Cross-validated variants: `fit_task_cv.py`, `decode_select_voxels_cv.py`.
Cluster submission helper: `encoding_model/cluster_scripts/submit_decode.py`.
The decoded per-trial `E` (mean) and `sd` (uncertainty) feed the neural
behavioural models.

### Stage 5 — behavioural / cognitive models (`bauer` + Bambi)

The PMCM and alternatives, via
[`risk_experiment/cogmodels/fit_model.py`](risk_experiment/cogmodels/fit_model.py):

```bash
python risk_experiment/cogmodels/fit_model.py 12          # PMCM main model
python risk_experiment/cogmodels/fit_model.py klw         # alt A (shared prior, equal noise)
python risk_experiment/cogmodels/fit_model.py 52          # alt B (varying priors, equal noise)
python risk_experiment/cogmodels/fit_model.py 42          # alt C (shared prior, varying noise)
python risk_experiment/cogmodels/fit_model.py eu          # alt D (expected utility)
python risk_experiment/cogmodels/fit_model.py neural32    # PMCM + decoded sd regressors (Fig 5)
```

Model labels (a non-exhaustive map of the ones in the paper):

| Label | Model |
|---|---|
| `12` | **PMCM** main model (this is *the* model of the paper) |
| `klw` | Alternative A — shared prior, equal noise |
| `52` | Alternative B — varying priors, equal noise |
| `42` | Alternative C — shared prior, varying noise |
| `eu` | Alternative D — expected-utility model |
| `neural*` (e.g. `neural32`) | PMCM with decoded neural uncertainty (`sd`) as regressors |

The psychophysical probit models (RNP) are in
[`risk_experiment/cogmodels/fit_probit.py`](risk_experiment/cogmodels/fit_probit.py)
(e.g. `probit_full`, `probit_full_session`); the symbolic-experiment probit is in
`risk_experiment/symbolic_experiment/fit_probit.py`.

Each fit writes `derivatives/cogmodels/model-{label}_trace.netcdf`. Sampling is
expensive (NUTS) — on a cluster use `risk_experiment/run_batch.py`.

### Stage 6 — figures

See §4.

---

## 4. Figure reproduction

The six submission figures are assembled from **per-panel vector PDF
ingredients** produced by the scripts in
[`risk_experiment/figures/`](risk_experiment/figures/). Each script reads the
cached traces, derives the plotted quantities, **caches them to a source-data
TSV**, and renders the panel in the house style defined by
[`figures/style.py`](risk_experiment/figures/style.py). Re-styling never re-runs
the expensive sampling; pass `--recompute` to force it.

Output panels land in `risk_experiment/revision/figures/` (PDF + SVG) with
per-panel source-data tables in `.../figures/source_data/`.

### Generators and commands

| Fig | Generator | Command | Notes |
|---|---|---|---|
| **1** | `figure_01_behavior.py` | `python -m risk_experiment.figures.figure_01_behavior` | Panels B (mag×order) + C (psychophysics + RNP insets). 1A/1D are manual schematics. |
| **2** | `figure_02_pmcm.py` | `python -m risk_experiment.figures.figure_02_pmcm` | PMCM PPC (2A) + group/participant posteriors (2B/2C) from `model-12`. (`cogmodels/notebooks/figure2.ipynb` is the original exploratory version.) |
| **3** | `figure_03_alt_models.py` | `python -m risk_experiment.figures.figure_03_alt_models` | A–D = PPCs of the four alternative models (`klw`/`52`/`42`/`eu`). |
| **4** | `figure_04_model_comparison.py` | `python -m risk_experiment.figures.figure_04_model_comparison` | Standalone ELPD model comparison (former panel 3E). |
| **5** | `figure_05_neural.py` (+ manual 5A) | `python -m risk_experiment.figures.figure_05_neural` | Decoding (5B) + neural→behaviour 5C/5D (`model-neural32`). 5A nPRF brain renders are **manual** (`registration/make_figures.ipynb`). |
| **6** | `figure_06_symbolic.py` | `python -m risk_experiment.figures.figure_06_symbolic` | Panel B = symbolic psychophysics + RNP insets. 6A is a manual schematic. |

Add `--recompute` to any of the scripted figures to rebuild its source-data TSV
from the trace instead of using the cache.

### Verify the exports

After (re)generating panels, check them against the Nature Communications artwork
rules (page width 88/120/180 mm, height ≤ 230 mm, min font ≥ 6 pt, fonts embedded
as editable Type 42 text not paths, RGB not CMYK):

```bash
~/mambaforge/envs/risk7t/bin/python -m risk_experiment.figures.verify
# or for specific files:
~/mambaforge/envs/risk7t/bin/python -m risk_experiment.figures.verify Figure_1.pdf
```

### Final assembly (Affinity Designer)

The Python scripts emit each panel as a standalone vector PDF **already at its
final physical size**. Final multi-panel composites, panel letters, and the
manual panels are assembled in **Affinity Designer** — see
[`risk_experiment/revision/notes/AFFINITY_FIGURE_SIZES.md`](risk_experiment/revision/notes/AFFINITY_FIGURE_SIZES.md)
for canvas widths per figure and the golden rule (place panels at 100 %, never
rescale text below the 6 pt floor). Re-run `verify` on the exported
`Figure_N.pdf` afterwards.

### Manual (non-data) panels

These are **not** generated by the scripts and must be handled by hand:

- **Fig 1A** — trial-sequence schematic.
- **Fig 1D** — PMCM intuition cartoon.
- **Fig 5A** — nPRF preferred-numerosity brain renders (3T & 7T); raster ≥ 300 dpi,
  but keep colourbar ticks/labels as vector.
- **Fig 6A** — symbolic trial-sequence schematic.

(1A, 1D and 6A were the panels the editor flagged as "suspected third-party
content"; confirm they are author-created.)

### Old→new figure mapping

The scripts and the conceptual `README.md` predate the final renumbering. Map:

| New (published) | Was | Content |
|---|---|---|
| Figure 1 | Figure 1 | Task paradigm + behavioural order×stake effects |
| Figure 2 | Figure 2 | PMCM PPC + group/participant posteriors |
| Figure 3 | Figure 3 A–D | PPCs of the four alternative models |
| **Figure 4** | **Figure 3 E** | ELPD model comparison (split out to stand alone) |
| **Figure 5** | **Figure 4** | nPRF mapping + decoding + neural→behaviour |
| **Figure 6** | **Figure 5** | Symbolic experiment paradigm + psychophysics |

So: **old Fig 3E → new Fig 4; old Fig 4 → new Fig 5; old Fig 5 → new Fig 6.**

---

### Supplementary material

Two supplementary results are produced by notebooks (no `figure_0N` script):

| Supplement | Notebook |
|---|---|
| **Supplementary Fig. 1** — per-subject behavioural patterns + posterior-predictive plots for representative participants | `risk_experiment/cogmodels/notebooks/supplfigure1.ipynb` (loads `ses-{3t2,7t2}_model-1` traces) |
| **Supplementary Text 3** — symbolic-experiment model comparison (PMCM vs alternatives vs EU) | `risk_experiment/symbolic_experiment/notebooks/model_comparison_probit.ipynb` (`az.compare` over symbolic `model{0,1,2,3}`) |

Parameter-recovery (Supplementary Text 1/2) is
`cogmodels/notebooks/parameter_recovery.ipynb` (+ generator
`cogmodels/parameter_recovery.py`).

---

## 5. Reported statistics (p-values, confidence intervals)

Every *p*-value, correlation, and test statistic in the main text is recomputed
from the traces/data by a single script — the source of truth for the numbers:

```bash
~/mambaforge/envs/risk7t/bin/python risk_experiment/revision/report_statistics.py
```

It writes two files into `risk_experiment/revision/notes/`:

- **`reported_statistics.md`** — every statistic in **manuscript reading order**
  (grouped by the four Results subsections), each as a copy-paste-ready string in
  Nature Communications format (exact *p* to 3 dp / `< 0.001`; 95% CI on every
  correlation; Pearson df = n − 2; `p_Bayesian` kept distinct from frequentist *p*),
  next to the current manuscript value and a note on anything that differs.
- **`reported_statistics.csv`** — the same, machine-readable.

See [`revision/notes/STATISTICS_AUDIT.md`](risk_experiment/revision/notes/STATISTICS_AUDIT.md)
for the full reconciliation (what reproduces exactly, what came from superseded
fits, what needs the cluster). Two helpers feed it:

- **Per-session model refits** — the per-session PMCM traces that produced some
  manuscript per-session numbers were lost; regenerate them with
  `python risk_experiment/revision/refit_per_session.py` (writes to
  `derivatives/cogmodels/revision_refit/`, leaving the published traces untouched).
- Combined ("both"-session) PMCM values come from the published `model-12` /
  `model-neural32` traces and reproduce exactly.

> Note: the *Bayesian* p-values are MCMC tail probabilities, so the third decimal
> carries a little Monte-Carlo noise (~±0.003 near p ≈ 0.16); rerun for stability.

---

## 6. Quick path: figures only, from published traces

```bash
conda activate risk7t                         # env from environment.yml
# (ensure src/bauer @ e246d78 worktree is installed; see §1)
# (ensure /data/ds-risk/derivatives/cogmodels/model-*_trace.netcdf exist)

python -m risk_experiment.figures.figure_01_behavior
python -m risk_experiment.figures.figure_01D_model_schematic
python -m risk_experiment.figures.figure_02_pmcm
python -m risk_experiment.figures.figure_03_alt_models
python -m risk_experiment.figures.figure_04_model_comparison
python -m risk_experiment.figures.figure_05_neural
python -m risk_experiment.figures.figure_06_symbolic
python -m risk_experiment.figures.verify        # check all exported panels

# Reported statistics (every p-value / CI, manuscript order):
python risk_experiment/revision/report_statistics.py

# Manual panels (1A, 5A brain renders, 6A) + final multi-panel assembly:
# Affinity Designer, per revision/notes/AFFINITY_FIGURE_SIZES.md.
```

---

## 7. What is **not** needed to reproduce the paper (`archive/`)

Exploratory / infrastructure code that is **not** part of the six figures or the
reported statistics has been moved to [`archive/`](archive/) (paths preserved;
every move is a reversible `git mv`). You can ignore it for reproduction. See
[`archive/CLEANUP_NOTES.md`](archive/CLEANUP_NOTES.md) for the full list and
rationale. Currently archived: a broken scratch script (`cogmodels/plot_ppc.py`),
TMS-targeting code (no TMS data in this paper), scheduling spreadsheets
(`session_lists/`), and the exploratory pupillometry / subcortical-ROI /
physiology (RETROICOR) pipelines (`pupil/`, `roi_analysis/`, `physiology/` — none
exercised by the six main figures).

Still in the tree: the `pupil*` / `subcortical_*` / `certainty*` model branches
inside `cogmodels/fit_model.py` / `fit_probit.py` (harmless, unused for the paper;
left to avoid editing the core fit scripts).
