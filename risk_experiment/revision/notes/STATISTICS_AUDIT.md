# Statistics audit — NCOMMS-24-63995B (de Hollander et al.)

Reproducibility audit of every statistic in the main text, for the Nature
Communications editorial pass (exact *p* to 3 decimals; `< 0.001` rule; **95% CI
for every correlation**).

**Single source of truth:** `risk_experiment/revision/report_statistics.py`
(run under the `risk7t` env → pinned bauer 0.1.0 @ `src/bauer`). It recomputes
everything below and writes `notes/reported_statistics.csv`.

Per-session PMCM traces were refit by `risk_experiment/revision/refit_per_session.py`
→ `derivatives/cogmodels/revision_refit/` (originals lost; existing combined
traces untouched).

---

## A. Reproduces — use these (3-dp / CI formatted)

| Manuscript location | Manuscript | Current value | Source |
|---|---|---|---|
| Prior mean risky vs safe, **both** | p_Bayesian = 0.16 | **0.156** | model-12 (session-avg) |
| Evidence noise 1st vs 2nd, both | p_Bayesian < 0.001 | **< 0.001** | model-12 |
| Neural reg, v1 (1st-option noise), both | p_Bayesian = 0.009 | **0.009** | model-neural32 |
| Neural reg, SD prior safe payoffs, both | p_Bayesian = 0.004 | **0.004** | model-neural32 |
| Psychophysical slope hi vs lo unc, 3T | p_Bayesian = 0.0458 | **0.046** | probit_neural4 (3t2) |
| Psychophysical slope hi vs lo unc, 7T | p_Bayesian = 0.0238 | **0.024** | probit_neural4 (7t2) |
| RNP distance hi vs lo unc, 3T | p_Bayesian = 0.012 | **0.012** | analyze_neural_probit.ipynb |
| RNP distance hi vs lo unc, 7T | p_Bayesian = 0.013 | **0.013** | " |
| RNP distance hi vs lo unc, both | p_Bayesian < 0.001 | **< 0.001** | " |
| Decoding accuracy (actual~decoded), 3T | r=0.176, t(29)=7.1, p<0.001 | **r = 0.176, 95% CI [0.130, 0.230], t(29)=7.1, p < 0.001** | get_decoding_info, per-run log |
| Decoding accuracy, 7T | r=0.16, t(29)=8.0 | **r = 0.165, 95% CI [0.120, 0.210], t(29)=8.0, p < 0.001** | " |
| Decoded uncertainty ~ error, 3T | r=0.33, t(29)=9.8 | **r = 0.332, 95% CI [0.260, 0.400], t(29)=9.8, p < 0.001** | " |
| Decoded uncertainty ~ error, 7T | r=0.38, t(29)=18.8 | **r = 0.379, 95% CI [0.340, 0.420], t(29)=18.8, p < 0.001** | " |
| Decoding-corr ~ v1, 3T | r(29)=-0.48, p=0.007 | **r = -0.500, 95% CI [-0.730, -0.170], p = 0.005** | brainbehavior.ipynb; pooled corr(n1,E) + ses-model-1 v1 |
| Decoding-corr ~ v1, 7T | r(29)=-0.42, p=0.021 | **r = -0.406, 95% CI [-0.670, -0.050], p = 0.026** | " |
| Mean decoded SD ~ v1, 3T | r(29)=0.44, p=0.014 | **r = 0.461, 95% CI [0.120, 0.700], p = 0.010** | brainbehavior.ipynb; mean sd + ses-model-1 v1 |
| Mean decoded SD ~ v1, 7T | r(29)=0.38, p=0.039 | **r = 0.406, 95% CI [0.050, 0.670], p = 0.026** | " |

> **#5/#6 method (recovered from `brainbehavior.ipynb`, deleted; commit b905bde):**
> the "decoding correlation" here is the **pooled raw** `pingouin.corr(n1, E)`
> over all trials (mask=npcr, n_voxels=0) — a *different* measure from the
> reported decoding accuracy (#3, which is per-run `corr(E, log(n1))` averaged).
> ν₁ = posterior-mean `n1_evidence_sd` from `ses-{session}_model-1`. Both the
> current `/data` traces and the fresh refit reproduce the manuscript.
| Symbolic: consistency ~ indifference | r(57)=0.34, p=0.00822 | **r = 0.344, 95% CI [0.09, 0.55], p = 0.008** | analyze_probit_models.ipynb (model0) |
| PMCM param test-retest (6 params) | r(29) 0.41–0.76, all p<0.05 | **range 0.46–0.76, all p<0.05** | refit model-1 (3t2 vs 7t2) |
| Symbolic stake×order RNP difference | p_Bayesian < 0.05 (summary) | summary; flag | — |

Frequentist correlation *p* to 3 dp: 0.00822 → **0.008**, 0.007, 0.021, 0.014,
0.039 (already 3 dp). House-style: Nature italicises/capitalises frequentist *P*
— author decision, not auto-applied.

---

## B. DOES NOT reproduce — author decision needed

Verified across the local Jun-2024 trace, the cluster May-2024 trace, **and a
fresh refit with the pinned analysis-era bauer 0.1.0** — all agree with each
other and disagree with the manuscript. The manuscript per-session PMCM numbers
came from an earlier model/code state that no longer survives.

| Manuscript location | Manuscript | Refit (current) | Effect on claim |
|---|---|---|---|
| Prior mean risky vs safe, 3T | p_Bayesian = 0.16 | **0.446** | none — still n.s. ("did not have higher mean") |
| Prior mean risky vs safe, 7T | p_Bayesian = 0.37 | **0.628** | none — still n.s. |
| Neural reg v1, 3T | p_Bayesian = 0.02 | **0.043** | model-neural32 session decomposition |
| Neural reg v1, 7T | p_Bayesian = 0.04 | **0.044** | " |
| Neural reg SD prior safe, 3T | p_Bayesian = 0.02 | **0.042** | " |
| Neural reg SD prior safe, 7T | p_Bayesian = 0.01 | **0.016** | " |

Per-session neural p-values come from the **neural32** session decomposition
(3T = `sd`; 7T = `sd` + `sd:session[T.7t2]`) — the same model used for the
combined value and Fig 5D — not from per-session `neural3` fits (whose
`risky_prior_std`/`safe_prior_std` regressor-name typo means they never receive
a `safe_prior_sd` neural slope). The combined ("both") values reproduce exactly;
the 3T/7T point values differ from the manuscript's (superseded older fit).

\* `neural3` regressor dict uses keys `risky_prior_std`/`safe_prior_std`, but
the parameters are named `..._sd`; bauer 0.1.0 silently drops the mismatched
keys, so `safe_prior_sd` never receives a neural regressor. The manuscript
per-session `safe_prior_sd` neural p-values therefore cannot have come from
`neural3` and are not reproducible.

The combined ("both") values for these same effects **do** reproduce (Section A),
and the overall claims rest on the combined results — so the headline findings
stand. Open question: how to report / whether to drop the per-session breakdowns.

---

## C. Not ported — need surface nPRF derivatives

| Manuscript | Source |
|---|---|
| Vertex-wise R² test-retest: r=0.25, t(29)=6.2, p<0.001 | per-subject corr of vertex R² between sessions; needs surface nPRF fits |
| Preferred-numerosity test-retest: r=0.15, t(29)=6.0, p<0.001 | source notebook not located |

These need the per-vertex nPRF R²/μ maps per session; not reconstructable from
the behavioural/decoding artifacts. 95% CIs computable once the per-subject r
vectors are regenerated.

---

## D. Leave alone (per brief)
- All `p < 0.001` — correct representation.
- `p_Bayesian < 0.05` used as a classification criterion (Fig 2C "almost all
  participants"; 5/58 and 13-participant counts) — thresholds, not single tests.
