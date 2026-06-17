# Figure caption brief — NCOMMS-24-63995B (de Hollander et al.)

For the caption/legend-writing agent. One entry per main figure: what each panel
shows, the conditions plotted, and **what the error/uncertainty represents**
(captions must state this). Panels marked *(manual)* are assembled in Affinity
(schematics / brain renders) — they still need a caption sentence even though no
script generates them.

## Terminology
- **"Figure legend" = "figure caption"** = the descriptive text block under the
  figure (Nature's term is "figure legend"). The little colour/marker key *inside*
  a panel is the "key", a different thing — don't conflate them in prose.
- Per-panel **source data** lives in `revision/figures/source_data/<panel>.tsv`
  (one tidy table per panel) if a caption needs to name the plotted quantity.

## House facts (true across figures)
- Payoff **ratio = n_risky / n_safe**. The risky option pays with probability
  **0.55**, so the **risk-neutral ratio = 1/0.55 ≈ 1.82** (expected values equal);
  dashed vertical reference lines mark it.
- **RNP** = risk-neutral point (the indifference probability). **RNP = 0.55 is
  risk-neutral**; RNP > 0.55 = risk-seeking, RNP < 0.55 = risk-averse.
- **Order** = presentation order of the two options: "Safe first" / "Risky first".
- Sessions: **3T** and **7T**; the risky-choice task is the `*2` session.
- Cohort: **sub-24 always excluded** (claustrophobia, one session only); some
  analyses additionally drop **sub-03** as an outlier. **Confirm exact n** with the
  authors before finalising each caption.
- These are *form-only* regenerations: **no statistic changed** from the published
  analysis.

---

## Figure 1 — Task paradigm & behavioural order × stake effects
- **1A** *(manual)* — trial-sequence schematic of the risky-choice task.
- **1B** — P(risky choice) vs safe-offer magnitude; hue = Order. Error bars =
  **±1 SEM across participants**. Dashed line at 0.5 = indifference.
- **1C** — psychophysical curves: P(risky choice) vs risky/safe payoff ratio, one
  column per stake tertile (Small / Medium / Large); hue = Order. **Curve error =
  ±1 SEM across participants.** Directly below each curve: the risk-neutral point
  (RNP) per Order, drawn as **the 95% HDI of the group-level `probit_full`
  posterior** (NB: different uncertainty type from the curve above — say so).
- **1D** *(manual / schematic)* — PMCM intuition cartoon: each option's magnitude
  is a precision-weighted Bayesian estimate (noisy likelihood + shared prior); the
  noisier (first-presented) option is pulled further toward the prior mean, and
  which option is pulled swaps with presentation order, shifting the EV comparison.

## Figure 2 — The PMCM model (combined 3T + 7T, model-12)
The combined estimate is the session-mean (3T level + ½·7T offset, softplus link).
- **2A** — PMCM posterior-predictive: points = group-mean data by Order, line +
  **95% HDI band** = posterior prediction; per stake size.
- **2B** — group-level parameter posteriors as filled KDEs: Evidence SD, Prior μ,
  Prior σ; hue = option (Option 1 / Option 2 for evidence; Safe / Risky for priors).
- **2C** — participant-level parameter *differences* (Option 1 − 2; Risky − Safe):
  per-subject **mean + 95% HDI**, points coloured by the participant's combined
  risk profile (risk-seeking / averse / neutral).

## Figure 3 — Posterior-predictive checks, four alternative models
4 × 3 grid: **rows = models**, **columns = stake size** (Small/Medium/Large).
Models: **A** = shared prior, equal noise; **B** = varying priors, equal noise;
**C** = shared prior, varying noise; **D** = expected-utility. In each panel:
points = data by Order, line = posterior-predictive mean, shaded = **95% HDI**;
dashed vertical = risk-neutral ratio. (Models that cannot separate the two orders
draw their prediction in grey; Model C keeps the Order colours.)

## Figure 4 — Model comparison (ELPD)
Standalone PSIS-LOO comparison over the five behavioural models (PMCM + A–D),
sorted best-first. Open circles = **ELPD ± SE**; grey triangles = **ELPD
difference ± dSE** vs the best model; dashed line at the best model's ELPD.
Reproduces `arviz.compare` / `plot_compare`.

## Figure 5 — nPRF decoding & neural-uncertainty effects
- **5A** *(manual brain renders)* — nPRF preferred-numerosity surface maps (3T & 7T).
- **5B** — per-participant decoding correlations, x = Scanner (3T / 7T):
  (i) decoded posterior mean vs objective log-numerosity (decoding accuracy),
  (ii) decoded posterior SD vs absolute decoding error (uncertainty calibration).
  Swarm = participants; white diamond = **mean ± SEM**; dashed line at r = 0.
- **5C** — posterior densities of the group-mean distance-from-risk-neutral, low vs
  high neural uncertainty (combined `probit_neural9`). Annotated *p* = posterior
  P(low-uncertainty distance > high-uncertainty distance).
- **5D** — group-level posteriors of the **neural (decoded-SD) regressor** on
  Evidence SD / Prior mean / Prior std; hue = option. Annotated *p* = posterior
  P(slope < 0) for the manuscript-reported coefficient (Evidence SD: Option 1,
  p ≈ 0.009; Prior std: Safe, p ≈ 0.004).

## Figure 6 — Symbolic (Arabic-numeral) experiment
- **6A** *(manual)* — symbolic trial-sequence schematic.
- **6B** — psychophysical curves per stake bin (five `n_safe` bins: 5–7, 7–9, 9–14,
  14–19, 19–28); hue = Order. Points = data, line + **95% HDI band** = probit
  posterior-predictive. Directly below each: the raw RNP per Order (**mean + 95%
  HDI**), positioned on the same axis so it reads against the curve's crossover.
