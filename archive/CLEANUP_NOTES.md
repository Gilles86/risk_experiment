# archive/CLEANUP_NOTES.md — conservative cleanup of non-paper analysis code

Goal: tidy dead / exploratory analysis code that did **not** end up in the
published paper, **without throwing anything away**. Nothing here is deleted.
The plan is to `git mv` dead files into `archive/`, preserving their sub-path
under the repo root, so every move is fully reversible (`git mv` back).

## Status

**Executed 2026-06-16** (a later session, `git mv` available). The following
non-paper code was archived into `archive/`, preserving its sub-path so every
move is reversible (`git mv` back). The package still imports and the figure
pipeline is intact (verified):

| Moved → `archive/...` | Why safe |
|---|---|
| `risk_experiment/cogmodels/plot_ppc.py` | Broken stub: imports `from tms_risk.cogmodels...` (wrong project) — would `ModuleNotFoundError` on load. Superseded by `cogmodels/utils.py:plot_ppc`. 0 references. |
| `risk_experiment/tms_targeting/` | TMS targeting from pRF centre-of-mass. **No TMS data in this paper.** 0 references anywhere. |
| `risk_experiment/session_lists/` | Scheduling spreadsheets (`week4.csv`, `week5.csv`), not analysis code. 0 references. |

**Also archived 2026-06-17** (author confirmed): exploratory pipelines absent
from the 6 main figures. No Python imports of any of them exist (verified), so
the package and figure pipeline still import cleanly:

| Moved → `archive/...` | Why safe |
|---|---|
| `risk_experiment/pupil/` | Pupillometry pipeline. Manuscript: eye-tracking "data not presented here". The `pupil*` model branches in `cogmodels/` read derivative files via `Subject`, not this dir. |
| `risk_experiment/roi_analysis/` | Subcortical ROI extraction (feeds exploratory `subcortical_*` models, not in the 6 figures). |
| `risk_experiment/physiology/` | RETROICOR QC + TAPAS artefacts. Methods note RETROICOR was *not* used as a final confound. (The `physiology_only` flag in `prepare/cleanup_raw_data.py` is an unrelated parameter name, not a reference to this dir.) |

Left in place: the `pupil*` / `subcortical_*` / `certainty*` **model branches**
inside `cogmodels/fit_model.py` / `fit_probit.py` — editing those core files
risks the figure/stat reproduction, and the branches are harmless (unused for the
paper). They simply have no in-repo script to regenerate their input derivatives.

---

## Proposed moves (verified safe; not yet executed)

| From | To | Reason |
|---|---|---|
| `risk_experiment/cogmodels/plot_ppc.py` | `archive/risk_experiment/cogmodels/plot_ppc.py` | Dead scratch script. Its first import is `from tms_risk.cogmodels.utils import plot_ppc, cluster_offers` — a **wrong-project import** (`tms_risk`, not `risk_experiment`) that would raise `ModuleNotFoundError` on load. It is a copy-paste stub, never imported by any module, notebook, or figure script, and is fully superseded by the working `plot_ppc()` function in `risk_experiment/cogmodels/utils.py`. |

To execute (human, shell access):

```bash
cd /Users/gdehol/git/risk_experiment
mkdir -p archive/risk_experiment/cogmodels
git mv risk_experiment/cogmodels/plot_ppc.py \
       archive/risk_experiment/cogmodels/plot_ppc.py
```

### Why this one is safe

- Grepped for references: the only tracked file that imports the *module path*
  `cogmodels.plot_ppc` is `analyze_model.py` (`from utils import plot_ppc`, an
  ambiguous relative import — not this module), and nothing else.
- The protected files (`figures/figure_0N_*.py`, `cogmodels/fit_model.py`,
  `cogmodels/fit_probit.py`, `utils/data.py`, `figures/style.py`,
  `figures/verify.py`) reference the `plot_ppc` **function in
  `cogmodels/utils.py`**, never this module — confirmed by grep.

---

## Candidates — NOT moved, need author confirmation

Conservative: each item below is plausibly dead/exploratory and **not** part of
the six main figures, but is either (a) wired into the data ecosystem, (b) a
standalone inspection tool that may still run, or (c) gitignored/submodule and
therefore out of scope for `git mv`. Left in place pending author sign-off.

### A. Standalone analysis/inspection scripts (uncertain — leave for now)

| Path | Note |
|---|---|
| `risk_experiment/cogmodels/analyze_model.py` | Posterior inspection for `fit_model.py` traces. Imported nowhere; `from utils import plot_ppc` is ambiguous. Likely run interactively. Not in the figure pipeline, but not broken like `plot_ppc.py`. |
| `risk_experiment/cogmodels/analyze_probit.py` | Same, for `fit_probit.py`. Imported nowhere; likely interactive inspection. |
| `risk_experiment/cogmodels/parameter_recovery.py` | Parameter-recovery simulation driver. Not imported; supports the (supplementary) recovery notebooks. |

### B. Exploratory analyses absent from the 6 main figures (but wired into `Subject`)

These feed the *alternative / neural-regressor* models (`pupil*`,
`subcortical_*`, `certainty*`, `roi_analysis`) that exist in
`cogmodels/fit_model.py` / `fit_probit.py` but are **not** among the six main
figures. They are reachable via methods on the `Subject` class in
`utils/data.py` (e.g. `get_pupil`, `get_roi_timeseries`), so moving them risks
breaking that data-access surface. **Leave; confirm with author whether these
were reported in supplementary material.**

| Dir | Files | Note |
|---|---|---|
| `risk_experiment/pupil/` | `preprocess.py`, `create_pupil_parquet.py`, `extract_pre_post_pupil.py`, `fit_pupil_model.py` | Pupillometry pipeline; feeds `pupil*` models. Not in main figures. |
| `risk_experiment/physiology/` | `plot_retroicor_r2.py` + `*.m` + `tapas_*.mat` | RETROICOR physiological-confound QC. The `.mat`/`.m` files are TAPAS artefacts. |
| `risk_experiment/roi_analysis/` | `extract_signal.py`, `extract_prestim_baseline.py`, `fit_glm_model.py` | Subcortical ROI extraction; feeds `subcortical_*` models. |
| `risk_experiment/tms_targeting/` | `make_volumetric_r2_com.py`, `sample_volumetric_com_to_surface.py` | TMS targeting from pRF centre-of-mass. **No TMS data in this paper** — strongest "exploratory" candidate, but harmless and self-contained; left for confirmation. |
| `risk_experiment/session_lists/` | `week4.csv`, `week5.csv` | Scheduling spreadsheets, not analysis code. |

### C. Out of scope for `git mv` (gitignored or submodule — do NOT archive)

Listed for completeness; these must **not** be moved.

| Path | Why excluded |
|---|---|
| `**/.ipynb_checkpoints/`, `**/*test*.ipynb` | Already **gitignored** (`.gitignore` lines `.ipynb_checkpoints`, `*test*.ipynb`) — untracked, so not `git mv`-able. This includes the scratch notebooks `pupil/test.ipynb`, `pupil/test copy.ipynb`, `cogmodels/test_prestim.ipynb`, `cogmodels/notebooks/test.ipynb`, `symbolic_experiment/notebooks/test*.ipynb`. |
| `build/`, `scripts/bdist.*`, `lib/` | Build artefacts; gitignored. |
| `src/` (incl. `src/bauer`) | Editable source checkout (bauer 0.1.0 worktree); gitignored and explicitly protected by the brief. |
| `braincoder/` | Registered **git submodule** (`.gitmodules`). Moving it would break the submodule mapping. |
| `risk_experiment/revision/**` | Explicitly protected by the brief (revision tooling, BIDS fixers). |

---

## Files explicitly verified as STILL-REFERENCED → left untouched

Grepped before considering any move; all are imported by protected code and were
**not** moved or modified:

- `risk_experiment/cogmodels/fit_model.py` — imported by `figure_03`, `figure_04`.
- `risk_experiment/cogmodels/fit_probit.py` — imported by `figure_01`.
- `risk_experiment/cogmodels/utils.py` — imported by figures (the real `plot_ppc`).
- `risk_experiment/symbolic_experiment/fit_probit.py` — imported by `figure_06`.
- `risk_experiment/utils/data.py`, `risk_experiment/utils/__init__.py` — `Subject` + helpers, imported broadly.
- `risk_experiment/figures/style.py`, `figures/verify.py`, `figures/figure_0N_*.py` — the figure pipeline.

**No still-referenced code was moved.** (In fact, no code was moved at all this
pass — see Status above.)

## Dead/orphan code archived 2026-06-17 (minimization audit)

A dependency-graph audit (transitive closure from the figure scripts +
`report_statistics.py` + `fit_model`/`fit_probit`/`Subject`) confirmed the
following are imported by **nothing** in the reproduction or pipeline surface.
Moved to `archive/` (paths preserved); package + all figure modules still import.

| Moved → `archive/...` | Why |
|---|---|
| `risk_experiment/visualize/` | Orphan pycortex quicklook scripts; `visualize_surf.py`/`visualize_average.py` have a broken `from utils import get_alpha_vertex` (error on load). Fig 5A renders are manual. 0 importers. |
| `risk_experiment/cogmodels/model_recovery/` | Self-contained model-recovery sim subtree; notebooks have a stale `from risk_experiment.ln.fit` import. Not a paper figure. 0 importers. |
| `risk_experiment/cogmodels/analyze_model.py`, `analyze_probit.py` | Interactive posterior-inspection scripts, superseded by the figure scripts. 0 importers. |
| `risk_experiment/symbolic_experiment/fit_nlc.py` | Alt non-linear-cumulative model, unused in the paper. 0 importers. |
| `risk_experiment/utils/decode.py` | Orphan (the real decoder is `encoding_model/decode.py` via braincoder). 0 importers. |
| `risk_experiment/utils/run_batch.py` | **Empty file** (0 lines). |
| `risk_experiment/utils/gzip_all_niis.py` | One-off maintenance util. 0 importers. |
| `risk_experiment/figures/likelihood_prior_revision.ipynb` | Source notebook ported verbatim into `figures/figure_01D_model_schematic.py`; fully superseded. |

**Kept deliberately** (the minimization audit flagged them but they back
currently-cited statistics or are reproduction entry points): the analysis
notebooks `cogmodels/notebooks/analyze_neural_probit.ipynb` and
`neural_model_comparison.ipynb` (sources of the per-session slope / RNP-distance /
neural p-values cited in `revision/notes/STATISTICS_AUDIT.md`); `figure_02_pmcm.py`
and `figure_05_neural.py` (real figure scripts); `utils/argparse.py` (pulled in by
`utils/__init__`); `utils/surface.py` and `cogmodels/parameter_recovery.py`
(pipeline producers). Other scratch notebooks (`model_comparison.ipynb`,
`supplfigure1.ipynb`, `symbolic_experiment/figure1.ipynb`, the `notebooks/revision1/`
working notebooks) were left pending author confirmation.

## Scratch notebooks archived 2026-06-17

Notebooks superseded by the `figure_0N_*.py` scripts (each saved a figure the
script now produces). Archived:

| Moved → `archive/...` | Superseded by |
|---|---|
| `cogmodels/notebooks/model_comparison.ipynb` | `figures/figure_04_model_comparison.py` |
| `notebooks/revision1/ppcs.ipynb` | `figure_02_pmcm.py` / `figure_03_alt_models.py` |
| `notebooks/revision1/ppcs_symbolic.ipynb` | `figure_06_symbolic.py` |
| `notebooks/revision1/stake_plots.ipynb` | `figure_01_behavior.py` |
| `symbolic_experiment/figure1.ipynb` | `figure_06_symbolic.py` |

Deleted: `notebooks/revision1/rnps_neural_noise.ipynb` (empty, 0 bytes, untracked).
The now-empty `notebooks/` tree was removed.

**KEPT as necessary (sole source of a supplementary result; documented in
REPRODUCE.md §4 "Supplementary material"):**
- `cogmodels/notebooks/supplfigure1.ipynb` — Supplementary Figure 1.
- `symbolic_experiment/notebooks/model_comparison_probit.ipynb` — symbolic model
  comparison (Supplementary Text 3).

(`test*.ipynb` under both notebooks/ dirs are gitignored scratch — untracked, left
in place.)
