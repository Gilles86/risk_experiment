# Rapid Changes in Risk Attitudes Originate from Bayesian Inference on Parietal Magnitude Representations

**de Hollander, G., et al. (2024)**
*Nature Communications* (under revision)

Preprint: https://www.biorxiv.org/content/10.1101/2024.08.23.609296v1.full

This repository contains all analysis code for the above paper, which investigates how the brain's representations of numerical magnitude in parietal cortex shape moment-to-moment risk preferences.

---

## The Core Idea

**Why do people's risk preferences fluctuate from trial to trial?**

The paper proposes and tests a _Bayesian Perceptual Magnitude Comparison (PMC) model_: participants perceive the magnitudes of the risky and safe options through noisy parietal representations, form posterior beliefs about those magnitudes, and choose based on a comparison of those beliefs. Crucially, the uncertainty of the parietal representation — decoded from IPS fMRI activity on each trial — predicts trial-by-trial shifts in risk preferences.

**Key findings:**

1. Participants are systematically risk-seeking for small stakes and risk-averse for large stakes (a "stake effect"), captured by the PMC model's prior beliefs over magnitudes.
2. The PMC model (6 parameters) outperforms all alternative models (expected utility, shared priors, equal noise) in formal ELPD-LOO model comparison.
3. Trial-by-trial decoded neural uncertainty from parietal cortex predicts the noise parameter for the 1st-presented option (`n1_evidence_sd`), directly linking brain to behaviour.
4. The stake-dependent risk behaviour replicates in a purely symbolic (Arabic numeral) experiment, ruling out low-level perceptual confounds.

---

## Experimental Design

**Main experiment** (`/data/ds-risk/`):

- **N = 30** participants (sub-02 to sub-32, excluding sub-24).
- **Sessions**: 4 per participant — `ses-3t1`, `ses-3t2` (3T MRI) and `ses-7t1`, `ses-7t2` (7T MRI). Sessions ending in `1` are localiser runs; sessions ending in `2` are the main task.
- **Task**: Each trial shows a risky option (50/50 gamble, pays N or 0 CHF) then a safe option (fixed M CHF). Participants choose one. The presentation order (`risky_first`) is counterbalanced. Stimuli are dot arrays (non-symbolic numerical magnitude).
- **Runs per main session**: 8 runs (~192 trials/session, ~768 trials total across 3T+7T).
- **Localiser**: A separate number-discrimination run maps numerical pRFs in IPS/V5.

**Symbolic experiment** (`/data/ds-symbolicrisk/`):

- Independent sample, same task with Arabic numerals replacing dot arrays. No perceptual noise manipulation.

---

## Data Organisation (BIDS)

```
/data/ds-risk/
├── sub-{id}/
│   └── ses-{session}/
│       ├── func/          # BOLD fMRI runs (.nii.gz + .json)
│       ├── anat/          # T1w anatomicals (defaced)
│       └── beh/           # behavioural event files (.tsv per run)
└── derivatives/
    ├── fmriprep/          # fMRIPrep preprocessing output
    ├── cogmodels/         # fitted model traces (.netcdf) and summaries (.tsv)
    ├── encoding_model.cv.denoise.natural_space/
    │                      # per-voxel nPRF parameters (mu, sd, amplitude, baseline)
    └── decoded_pdfs.volume.cv_voxel_selection.denoise.natural_space/
                           # trial-by-trial decoded numerosity PDFs (mean E, sd per trial)
```

Behavioural event files contain per-trial columns: `n1` (risky amount), `n_safe`, `chose_risky`, `risky_first`, and after decoding: `E` (decoded numerosity estimate), `sd` (decoded uncertainty).

Key model trace files in `derivatives/cogmodels/`:

| File | Description |
|---|---|
| `model-12_trace.netcdf` | PMC model, `session` regressor on all 6 params (combined 3T+7T) |
| `model-neural32_trace.netcdf` | PMC model + decoded `sd` × `session` on all 6 params |
| `model-probit_full_session_trace.netcdf` | Psychophysical probit model, combined sessions |
| `model-probit_neural9_trace.netcdf` | Probit + `median_split_sd` × stake × order, combined sessions |
| `model-klw_trace.netcdf` | Alternative: shared prior, equal noise |
| `model-52_trace.netcdf` | Alternative: varying priors, equal noise |
| `model-42_trace.netcdf` | Alternative: shared prior, varying noise |
| `model-eu_trace.netcdf` | Alternative: Expected Utility model |

---

## Analysis Pipeline

### Step 1 — fMRI Preprocessing

Standard preprocessing via **fMRIPrep**: head motion correction, slice timing correction, spatial normalisation (MNI + subject T1w), surface reconstruction via FreeSurfer. Batch scripts are in `risk_experiment/cluster_preprocess/`.

### Step 2 — Numerical pRF Encoding Model (Figures 4A, 4B)

**Library**: [`braincoder`](https://braincoder-devs.github.io/)

Braincoder implements population receptive field (pRF) encoding models and their inversion (decoding) using gradient-based optimisation (Keras/Adam) and Bayesian inference.

**Key classes:**

| Class | Description |
|---|---|
| `GaussianPRF` | 1-D Gaussian pRF (parameters: `mu`, `sd`, `amplitude`, `baseline`) |
| `ParameterFitter` | Fits pRF parameters per voxel using Adam optimisation |
| `ResidualFitter` | Estimates inter-voxel noise covariance matrix Ω from model residuals |
| `StimulusFitter` | **Decoding**: inverts the fitted encoding model to recover a posterior PDF over stimulus (numerosity) from BOLD activity on each task trial |

**Pipeline** (`risk_experiment/encoding_model/`):

1. Fit nPRF parameters (`mu`, `sd`) to each voxel using the localiser run with `ParameterFitter.fit()`.
2. Estimate residual covariance Ω across voxels with `ResidualFitter.fit()`.
3. For each task trial, decode the presented numerosity: `StimulusFitter.fit()` recovers a posterior PDF — its mean (`E`) and standard deviation (`sd`) are saved per trial.

`Figure 4A`: Smoothed map of preferred numerosity (`mu`) across IPS — a brain image, not in Source Data.
`Figure 4B`: Correlation between `E` (decoded) and `n1` (actual) per subject/session, and between `sd` and absolute error.

### Step 3 — Psychophysical (Probit) Models (Figures 1C, 4C, 5B)

**Library**: [Bambi](https://bambinos.github.io/bambi/) (wraps PyMC)

A hierarchical probit regression characterises how the log-ratio of magnitudes drives choices. The key predictor is `x = log(n_risky / n_safe)`. Models include interactions with stake size (`C(n_safe)`), presentation order (`risky_first`), scanner session, and neural uncertainty (`median_split_sd`).

**Key models** (`risk_experiment/cogmodels/fit_probit.py`):

| Label | Formula (simplified) | Figure |
|---|---|---|
| `probit_full_session` | `chose_risky ~ x*risky_first*C(n_safe) + x*session + (…\|subject)` | Fig 1C RNP insets |
| `probit_neural9` | `chose_risky ~ x*risky_first*median_split_sd*C(n_safe) + x*risky_first*C(n_safe)*session + (…\|subject)` | Fig 4C |

The **Risk-Neutral Probability (RNP)** is extracted via `extract_intercept_gamma()` in `risk_experiment/cogmodels/utils.py`:

```
RNP = exp(intercept / gamma)
```

where `intercept` is the model intercept (log-ratio at indifference) and `gamma` is the slope. RNP < 0.5 → risk-seeking; RNP > 0.5 → risk-averse.

### Step 4 — PMC Computational Model (Figures 2, 3, 4D)

**Library**: [`bauer`](https://github.com/ruffgroup/bauer) (Bayesian Estimation of Perceptual, Numerical and Risky judgements)

Bauer implements a family of Bayesian perceptual models for choice tasks. It builds hierarchical [PyMC](https://www.pymc.io/) models and samples with NUTS.

**The PMC model** (`RiskModel` / `RiskRegressionModel`):

On each trial, the model forms a posterior belief about each option's magnitude by combining a Gaussian prior with noisy perceptual evidence (in log-space):

```
posterior(n̂) = Normal(
    mu = (prior_mu / prior_sd² + log(n) / evidence_sd²) / (1/prior_sd² + 1/evidence_sd²),
    sd = 1 / sqrt(1/prior_sd² + 1/evidence_sd²)
)
```

The probability of choosing the risky option is:

```
p(choose risky) = Φ((n̂₂_mu − n̂₁_mu) / sqrt(n̂₁_sd² + n̂₂_sd²))
```

Six parameters, each estimated hierarchically across subjects:

| Parameter | Meaning |
|---|---|
| `n1_evidence_sd` | Perceptual noise for the 1st-presented option |
| `n2_evidence_sd` | Perceptual noise for the 2nd-presented option |
| `risky_prior_mu` | Prior mean (log-space) for the risky option |
| `safe_prior_mu` | Prior mean (log-space) for the safe option |
| `risky_prior_sd` | Prior width for the risky option |
| `safe_prior_sd` | Prior width for the safe option |

**Hierarchical structure** (per parameter, softplus-transformed):

```
group_mu    ~ Normal(prior, sigma)        # group-level mean
group_sd    ~ HalfCauchy(0.25)            # group-level spread
offset[s]   ~ Normal(0, 1)               # per-subject offset
param[s]    = softplus(group_mu + group_sd * offset[s])
```

**Key model variants:**

| Label | Regressors | Figure |
|---|---|---|
| `model-12` | All 6 params × `session` (3T vs 7T) | Figs 2A–C |
| `model-klw` | Shared prior, equal noise (3 params) | Fig 3A |
| `model-52` | Varying priors, equal noise (4 params) | Fig 3B |
| `model-42` | Shared prior, varying noise (4 params) | Fig 3C |
| `model-eu` | Expected utility (no Bayesian inference) | Fig 3D |
| `model-neural32` | All 6 params × `sd` × `session` | Fig 4D |

**Session averaging** (Figs 2B/C): Model-12 estimates separate 3T and 7T parameters. To obtain a single combined-session estimate, posteriors are averaged over sessions:

```python
from bauer.utils.math import softplus_np
combined = softplus_np(Intercept + 0.5 * session[T.7t2])
```

**Model comparison** (Fig 3E): LOO expected log predictive density (ELPD) via `az.compare()`, computed after running `pm.compute_log_likelihood()`.

**Neural–behavioural link** (Fig 4D): `model-neural32` adds the decoded `sd` (trial-by-trial neural uncertainty) as a continuous regressor on each PMC parameter. A significantly positive slope on `n1_evidence_sd` (posterior p > 0.95) means higher neural uncertainty → higher perceptual noise for the first option, directly linking parietal representations to choice variability.

---

## Repository Structure

```
risk_experiment/
├── cogmodels/
│   ├── fit_model.py           # PMC model fitting (RiskModel / RiskRegressionModel)
│   ├── fit_probit.py          # Probit model fitting (Bambi)
│   ├── utils.py               # extract_intercept_gamma(), softplus helpers
│   ├── model_recovery/        # Parameter & model recovery simulations
│   └── notebooks/
│       ├── figure2.ipynb      # Figs 2A–C (PMC model results)
│       ├── model_comparison.ipynb
│       ├── neural_model_comparison.ipynb
│       ├── parameter_recovery.ipynb
│       ├── analyze_neural_probit.ipynb
│       └── supplfigure1.ipynb
├── encoding_model/
│   └── notebooks/
│       ├── analyze_decoding_natural_space.ipynb   # Fig 4B
│       └── analyze_encoding_cvr2.ipynb            # Fig 4A
├── figures/
│   └── likelihood_prior_revision.ipynb  # Fig 2D schematic
├── notebooks/
│   └── revision1/
│       ├── ppcs.ipynb          # PPCs (Figs 2A, 3A–D)
│       ├── ppcs_symbolic.ipynb
│       └── stake_plots.ipynb   # Fig 1B
├── symbolic_experiment/
│   ├── figure1.ipynb
│   └── notebooks/
│       ├── analyze_probit_models.ipynb
│       └── model_comparison_probit.ipynb
├── registration/
│   └── make_figures.ipynb      # Fig 4A brain maps (pycortex)
├── prepare/                    # BIDS preparation, event file creation
├── preproc/                    # fMRI surface smoothing
└── surface/                    # IPS mask creation, pRF centre of mass

paper/
├── deHollanderetal2024v3.pdf   # manuscript (latest version)
├── Source_Data.xlsx            # source data for all figures (Nature Comms)
├── create_source_data.py       # generates Source_Data.xlsx
└── plot_source_data.py         # verification plots → source_data_figures.pdf
```

---

## Installation

```bash
git clone https://github.com/Gilles86/risk_experiment.git
cd risk_experiment
pip install -e .
```

Also install the two core libraries:

```bash
pip install git+https://github.com/Gilles86/bauer.git
pip install git+https://github.com/Gilles86/braincoder.git
```

Key dependencies: PyMC, ArviZ, Bambi, nilearn, nibabel, pandas, numpy, matplotlib.

A Docker environment is available:

```bash
pip install docker-compose
docker-compose build
```

---

## Reproducing Figures

### Source Data Excel file

```bash
python paper/create_source_data.py   # requires /data/ds-risk/ access
python paper/plot_source_data.py     # → paper/source_data_figures.pdf
```

### Fitting models from scratch

```bash
# Main PMC model (combined sessions)
python risk_experiment/cogmodels/fit_model.py --model_label 12

# Neural PMC model
python risk_experiment/cogmodels/fit_model.py --model_label neural32

# Psychophysical probit model
python risk_experiment/cogmodels/fit_probit.py --model_label probit_full_session

# Neural probit model (Fig 4C)
python risk_experiment/cogmodels/fit_probit.py --model_label probit_neural9
```

Models are computationally intensive (NUTS sampling). On a cluster, use the batch scripts in `risk_experiment/run_batch.py`.

---

## Source Data (`paper/Source_Data.xlsx`)

| Sheet | Content |
|---|---|
| Figure 1B | Mean prop. risky by safe payoff and presentation order |
| Figure 1B (per subject) | Per-subject |
| Figure 1C | Psychophysical curves (prop. risky vs log risky/safe ratio) by stake and order |
| Figure 1C (per subject) | Per-subject |
| Figure 1C RNP insets | Group-level RNP posteriors (mean + 95% HDI) from `probit_full_session` |
| Figure 2A | PMC model PPC (model 95% HDI + empirical mean ± SEM) |
| Figure 2B | Group-level PMC parameter posteriors (combined 3T+7T sessions) |
| Figure 2C | Subject-level parameter differences + behavioural risk preference category |
| Figure 3A-D | Alternative model PPCs |
| Figure 3E | ELPD-LOO model comparison |
| Figure 4B | Decoding accuracy per subject/session |
| Figure 4C | Distance to risk neutrality by neural uncertainty level (`probit_neural9`) |
| Figure 4D | Neural uncertainty slopes per PMC parameter (`model-neural32`) |
| Figure 5B | Symbolic experiment psychophysical curves |
| Figure 5B (per subject) | Per-subject |

**Note**: Figure 4A (nPRF brain map) is not included in Source Data — the underlying maps are NIfTI files stored in `derivatives/encoding_model.cv.denoise.natural_space/`.
