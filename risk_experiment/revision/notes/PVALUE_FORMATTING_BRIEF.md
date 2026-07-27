# P-value formatting pass — NCOMMS-24-63995B

**Goal:** make every reported *p*-value in the manuscript main text and figure legends an **exact value to three decimal places**, per the Nature Communications editorial request. This is a numeric-formatting and consistency pass only — no statistics, values, or notation may be reinterpreted. Make minimal, targeted edits and produce a change log for author review.

---

## Inputs

- The manuscript as **.docx** (cleaner to parse and edit than the PDF — equations and superscripts survive).
- If available: the **analysis outputs** that produced these values (ArviZ/`bauer`/PyMC summaries, posterior sample files, the correlation/t-test results). You will need these to recover exact values for any *p* currently printed at fewer than three decimals (see the critical rule below). If they are not available, flag those cases for the author rather than guessing.

---

## The rules

1. **Three decimal places, exactly**, for every reported *p*-value: `p = 0.046`, `p = 0.024`, `p = 0.008`.
2. **Values below 0.001 are written `< 0.001`** — never `0.000`, never `0.0008`. If a value rounds to `0.000`, render it as `< 0.001`. Leave existing `p < 0.001` exactly as they are; do **not** try to expand them to an exact figure.
3. **Rounding:** standard round-half-up to three decimals (`0.0458 → 0.046`, `0.0238 → 0.024`, `0.00822 → 0.008`). Prefer recomputing from source over rounding an already-rounded number — see below.

### ⚠️ Critical correctness rule — do not pad zeros blindly

A value currently printed at **two** decimals (e.g. `p=0.02`) must **not** simply become `0.020`. The true value might be `0.024`, which would be misreported as `0.020`. So:

- **Over-precise values** (4+ decimals, e.g. `0.0458`, `0.00822`): safe to round down to 3 decimals directly.
- **Under-precise values** (2 decimals, e.g. `0.02`, `0.16`): you are *adding* precision, which you cannot invent. **Look up the exact value from the analysis outputs and format it to 3 decimals.** If you cannot locate the exact value, **leave it and flag it in the change log** for the author to supply — do not pad with zeros.

### ⚠️ Preserve the notation — do not conflate Bayesian and frequentist

The manuscript deliberately uses **two different symbols**:
- `p_Bayesian` (subscript "Bayesian") — posterior probability mass above/below zero, defined in Methods. This is the authors' intentional notation.
- plain `p` — frequentist *p* from t-tests on correlations.

**Only change the number of decimal places. Do not change which symbol is used, do not merge `p_Bayesian` into `P`/`p`, and do not run a blanket find-replace on `p=`** — that would destroy the Bayesian subscript and conflate the two. Touch the digits, nothing else.

> House-style note (do **not** auto-apply): Nature italicises and capitalises the frequentist *P*. Whether to convert plain `p` → italic *P* is an author decision and risks catching `p_Bayesian` by accident, so leave the symbol styling alone and just list the frequentist `p` occurrences in your log so the author can decide.

---

## Known instances to fix

Search the .docx for these (verify each in context — don't trust the list blindly, and report any *p*-values you find that aren't listed here):

**Safe to round to 3 decimals:**
- `p_Bayesian = 0.0458` (3T) → `0.046` and `p_Bayesian = 0.0238` (7T) → `0.024` — the psychophysical-slope / neural-uncertainty split.
- `r(57) = 0.34, p = 0.00822` → `p = 0.008` — symbolic-experiment consistency×attitude correlation.

**Need the exact value (currently 2 decimals — fetch from analysis, do NOT pad):**
- group prior mean, risky vs safe: `p_Bayesian = 0.16 … = 0.37 … both = 0.16`
- neural-regressor result, noisiness of first option: `3T: p_Bayesian = 0.02; 7T: = 0.04` (the `both: = 0.009` is already 3 dp — leave it)
- prior-on-safe-payoffs regressor: `3T: p_Bayesian = 0.02, 7T: = 0.01` (the `both: = 0.004` is already fine)

**Figure legends:** update any *p*-value in legend *prose* to 3 decimals under the same rules. Note that values rendered *inside* figure panels (e.g. the `p = 0.0085` printed in the Figure 4 / former-3E neural-regressor panel) are part of the figure artwork and must be corrected during figure regeneration, not in the .docx — but ensure the legend prose and the in-panel value agree (both should read `0.009`, matching the `both: = 0.009` in the main text).

---

## Leave alone (do not "fix" these)

- All `p < 0.001` — this is the correct representation of sub-0.001 values, not something to expand.
- `p_Bayesian < 0.05` used as a **classification criterion / definition** (e.g. defining which participants count as having significantly noisier first-option evidence, or the 5/58 and 13-participant counts). These are thresholds, not single reported tests, so a cutoff is appropriate.
- All effect sizes, test statistics, degrees of freedom, ELPDs, credible intervals — out of scope for this pass.

## One judgment call — flag, don't change

- The summary `p_Bayesian < 0.05` for "a significant difference in RNP between the two presentation orders for all stake sizes" covers multiple tests at once. Strictly, exact values are preferred; practically, an editor will likely accept the summary. **Flag it for the author** with a note that exact per-stake-size values could be substituted if desired — do not auto-edit.

---

## Process & deliverables

1. Work on the .docx directly; make the smallest possible edit at each site (change digits only).
2. Produce a **change log**: a table of every edit as `location/context → old → new`, plus a separate list of any value you could not resolve (under-precise with no source found) and any frequentist `p` occurrences (for the house-style decision).
3. Do not alter surrounding text, formatting, tracked changes, or notation.

## Verification

After editing, scan the whole document and assert that **every** *p*-value matches one of these forms:
- `= 0.ddd` (exactly three decimals), or
- `< 0.001`, or
- a flagged threshold/criterion on the leave-alone list.

A regex such as `p(_Bayesian)?\s*[=<]\s*0?\.\d+` will surface every occurrence; report any that are not 3-decimal, not `< 0.001`, and not on the leave-alone list. List anything still outstanding.

## Do not

- Do not pad a 2-decimal value with zeros to fake 3-decimal precision — fetch the real value or flag it.
- Do not change `p_Bayesian` to `P`/`p`, or run a blanket `p=` replacement.
- Do not expand `p < 0.001` into an exact number.
- Do not touch values inside figure artwork (handled in figure regeneration) — only legend prose.
- Do not change any statistic, effect size, CI, or wording.
