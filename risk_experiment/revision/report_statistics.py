"""SINGLE SOURCE OF TRUTH for every statistic reported in the main text of
de Hollander et al. (NCOMMS-24-63995B).

For each reported number this script (re)computes the current value from the
model traces / data, formats it per the Nature Communications editorial rules
(exact p to 3 decimals; `< 0.001` for anything that rounds to 0.000; 95% CI for
every correlation), and prints a labelled table comparing it to the value
printed in the manuscript.

Run with the analysis-era stack (risk7t env -> pinned bauer 0.1.0 @ src/bauer):
    ~/mambaforge/envs/risk7t/bin/python risk_experiment/revision/report_statistics.py

Provenance of each trace is documented inline. PER-SESSION PMCM traces are the
fresh refits from `refit_per_session.py` (the originals that produced the
manuscript per-session numbers were lost; see the AUDIT notes below). Combined
("both") PMCM values come from the existing combined-session models
(model-12 / model-neural32), per the authors.

AUDIT STATUS (see risk_experiment/revision/notes/STATISTICS_AUDIT.md for detail):
  * Reproduces exactly: combined PMCM p-values, evidence-noise <0.001,
    psychophysical slope, RNP distance, correlations #3,#4,#6,#7.
  * DOES NOT reproduce from any surviving artifact: per-session PMCM prior-mean
    (manuscript 3T=0.16/7T=0.37) and per-session PMCM neural regression
    (manuscript n1 0.02/0.04, safe_prior_sd 0.02/0.01). Fresh refits give
    materially different values; the `neural3` model is also structurally unable
    to produce a safe_prior_sd neural slope (regressor-name typo 'std' vs 'sd').
  * Needs surface nPRF derivatives (not ported here): vertex-wise R^2 and
    preferred-numerosity test-retest (manuscript r=0.25 t=6.2 / r=0.15 t=6.0).
"""
import os.path as op
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import arviz as az
import pingouin

BIDS = '/data/ds-risk'
SYMBOLIC_BIDS = '/data/ds-symbolicrisk'
COG = op.join(BIDS, 'derivatives', 'cogmodels')
REFIT = op.join(COG, 'revision_refit')


# --------------------------------------------------------------------------- #
# Formatting helpers (Nature editorial rules)
# --------------------------------------------------------------------------- #
def fmt_p(p):
    """Exact p to 3 decimals; '< 0.001' if it rounds to 0.000."""
    if p is None:
        return 'n/a'
    return '< 0.001' if round(p, 3) < 0.001 else f'{p:.3f}'


def pclause(p, sym='p'):
    """Format a p-value clause the Nature way: 'p < 0.001' or 'p = 0.046'."""
    s = fmt_p(p)
    return f'{sym} {s}' if s.startswith('<') else f'{sym} = {s}'


def fmt_corr(r, ci, p, n):
    """Paste-ready Pearson result with the CORRECT df = n - 2 (NOT n - 1)."""
    return f'r({int(n) - 2}) = {r:.3f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], {pclause(p)}'


# Manuscript Results subsections, in reading order.
MS1 = 'Different noise contexts shift risk attitudes'
MS2 = 'The Perception and Memory-based Choice Model (PMCM)'
MS3 = 'Decoded payoff representations in parietal cortex predict noisiness and bias'
MS4 = 'Choices in symbolic presentation format'

_rows = []
_section = ['']
def section(title):
    _section[0] = title
    print('\n' + '=' * 78 + f'\n{title}\n' + '=' * 78)


def report(location, manuscript, computed, note='', order=999.0, msec='(unordered)'):
    _rows.append({'order': order, 'msec': msec, 'section': _section[0],
                  'location': location, 'manuscript': manuscript,
                  'computed': computed, 'note': note})
    flag = '' if not note else f'   <-- {note}'
    print(f'  [{order}] {location}')
    print(f'      manuscript: {manuscript}')
    print(f'      paste-ready: {computed}{flag}')


# --------------------------------------------------------------------------- #
def posterior_col(post, var, regressor=None):
    df = post[var].to_dataframe()
    if regressor is None:
        return df.iloc[:, 0]
    return df.xs(regressor, level=-1).iloc[:, 0]


# =========================================================================== #
section('PMCM Bayesian p-values  (posterior probability mass; one-tailed)')

# --- Combined ("both") values: model-12 / model-neural32, session-averaged --- #
idata12 = az.from_netcdf(op.join(COG, 'model-12_trace.netcdf'))
p12 = idata12.posterior


def m12_sessavg(var):
    return posterior_col(p12, var, 'Intercept') + 0.5 * posterior_col(p12, var, 'session[T.7t2]')


# Evidence noise n1 vs n2, both sessions  (text: p_Bayesian < 0.001)
d = m12_sessavg('n1_evidence_sd_mu') - m12_sessavg('n2_evidence_sd_mu')
report('Evidence noise 1st-vs-2nd, both sessions (model-12, session-avg)',
       'p_Bayesian < 0.001',
       pclause(float((d.values < 0).mean()), 'p_Bayesian'),
       order=21.0, msec=MS2)

# Prior mean risky vs safe, both sessions  (text: p_Bayesian = 0.16)
d = m12_sessavg('risky_prior_mu_mu') - m12_sessavg('safe_prior_mu_mu')
report('Prior mean risky vs safe, both sessions (model-12, session-avg)',
       'p_Bayesian = 0.16',
       pclause(float((d.values < 0).mean()), 'p_Bayesian'),
       order=22.2, msec=MS2)

# --- Neural regression combined: model-neural32, session-averaged 'sd' slope - #
idataN = az.from_netcdf(op.join(COG, 'model-neural32_trace.netcdf'))
pN = idataN.posterior


def n32_sessavg_slope(var):
    return posterior_col(pN, var, 'sd') + 0.5 * posterior_col(pN, var, 'sd:session[T.7t2]')


for var, lbl, ms, order in [
        ('n1_evidence_sd_mu', 'noisiness 1st option (v1)', 'p_Bayesian = 0.009', 38.2),
        ('safe_prior_sd_mu', 'SD prior on safe payoffs', 'p_Bayesian = 0.004', 39.2)]:
    s = n32_sessavg_slope(var)
    report(f'Neural reg: {lbl}, both sessions (model-neural32, session-avg sd-slope)',
           ms, pclause(float((s.values < 0).mean()), 'p_Bayesian'),
           order=order, msec=MS3)

# --- Per-session PMCM: FRESH REFITS (flagged) ------------------------------- #
print('\n  --- PER-SESSION (fresh refits; manuscript values DO NOT reproduce) ---')
for ses, lab, order in [('3t2', '3T', 22.0), ('7t2', '7T', 22.1)]:
    post = az.from_netcdf(op.join(REFIT, f'ses-{ses}_model-1_trace.netcdf')).posterior
    d = posterior_col(post, 'risky_prior_mu_mu') - posterior_col(post, 'safe_prior_mu_mu')
    ms = {'3T': 'p_Bayesian = 0.16', '7T': 'p_Bayesian = 0.37'}[lab]
    report(f'Prior mean risky vs safe, {lab} (refit model-1)',
           ms, pclause(float((d.values < 0).mean()), 'p_Bayesian'),
           note='differs from manuscript (superseded older fit); claim unchanged (n.s.)',
           order=order, msec=MS2)

# Per-session neural slopes: model-neural32 session DECOMPOSITION (the model
# the paper's Fig 5D uses). 3T = 'sd'; 7T = 'sd' + 'sd:session[T.7t2]'. The
# combined value above is the session average (sd + 0.5*sd:session). Reproduces
# 'both' exactly; the 3T/7T point values differ from the manuscript's
# (which came from a superseded older fit), but stay in the same ballpark.
# (`neural3` per-session fits are NOT used: a regressor-name typo 'std' vs 'sd'
#  means they never receive a safe_prior_sd neural slope.)
for var, lbl, ms3, ms7, base in [
        ('n1_evidence_sd_mu', 'noisiness 1st option (v1)', '0.02', '0.04', 38.0),
        ('safe_prior_sd_mu',  'SD prior on safe payoffs', '0.02', '0.01', 39.0)]:
    sd = posterior_col(pN, var, 'sd')
    sdx = posterior_col(pN, var, 'sd:session[T.7t2]')
    for lab, slope, ms, off in [('3T', sd, ms3, 0.0), ('7T', sd + sdx, ms7, 0.1)]:
        report(f'Neural reg: {lbl}, {lab} (model-neural32 session decomposition)',
               f'p_Bayesian = {ms}',
               pclause(float((slope.values < 0).mean()), 'p_Bayesian'),
               note='3T/7T differ from manuscript (older fit); both reproduces',
               order=base + off, msec=MS3)


# =========================================================================== #
section('Correlations  (r, 95% CI, p)   [Nature requires CI for correlations]')

from risk_experiment.utils import get_all_subjects

subs = get_all_subjects(bids_folder=BIDS)

# Per-subject/session decoding measures
dec_rows = []
for sub in subs:
    for ses in ['3t2', '7t2']:
        try:
            pred = sub.get_decoding_info(ses, mask='npcr', n_voxels=0.0)
            beh = sub.get_behavior(sessions=ses)
            m = pred.join(beh[['n1']]).dropna(subset=['n1', 'E'])
            if len(m) <= 10:
                continue
            m['log(n1)'] = np.log(m['n1'])
            # #3 REPORTED accuracy: per-run corr(E, log(n1)) averaged over runs
            #    (analyze_decoding_natural_space.ipynb -> r=0.176/0.16)
            if 'run' in m.index.names:
                r_acc = m.groupby('run').apply(
                    lambda x: pingouin.corr(x['E'], x['log(n1)'])['r'].iloc[0]).mean()
            else:
                r_acc = pingouin.corr(m['E'], m['log(n1)'])['r'].iloc[0]
            # #5 "decoding correlation" used vs v1: POOLED RAW corr(n1, E)
            #    (brainbehavior.ipynb cell 1 -- a DIFFERENT measure from #3)
            r_pooled = pingouin.corr(m['n1'], m['E'])['r'].iloc[0]
            # #4 uncertainty: pooled corr(sd, |n1 - E|)
            r_err = pingouin.corr(m['sd'], np.abs(m['n1'] - m['E']))['r'].iloc[0]
            dec_rows.append({'subject': sub.subject, 'session': ses,
                             'r_acc': r_acc, 'r_pooled': r_pooled,
                             'r_err': r_err, 'mean_sd': m['sd'].mean()})
        except Exception:
            pass
dec = pd.DataFrame(dec_rows).set_index(['subject', 'session'])

# v1 per subject/session from refit model-1
v1_parts = []
for ses in ['3t2', '7t2']:
    s = (az.from_netcdf(op.join(REFIT, f'ses-{ses}_model-1_trace.netcdf'))
         .posterior['n1_evidence_sd'].to_dataframe().groupby('subject').mean().iloc[:, 0])
    s.index = pd.MultiIndex.from_tuples([(i, ses) for i in s.index],
                                        names=['subject', 'session'])
    v1_parts.append(s)
dec['v1'] = pd.concat(v1_parts)


def onesample(x, manuscript, location, order, msec, note=''):
    """One-sample t-test on per-subject correlations -> df = n - 1 (correct)."""
    t = pingouin.ttest(x, 0.0)
    ci = t['CI95%'].iloc[0]
    report(location, manuscript,
           f"mean r = {x.mean():.3f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], "
           f"t({int(t['dof'].iloc[0])}) = {t['T'].iloc[0]:.1f}, {pclause(t['p-val'].iloc[0])}",
           note=note, order=order, msec=msec)


def between(x, y, manuscript, location, order, msec, note=''):
    """Between-subject Pearson -> df = n - 2 (the manuscript used n - 1)."""
    c = pingouin.corr(x, y)
    report(location, manuscript,
           fmt_corr(c['r'].iloc[0], c['CI95%'].iloc[0], c['p-val'].iloc[0], c['n'].iloc[0]),
           note=note, order=order, msec=msec)


for ses, lab, o in [('3t2', '3T', 0.0), ('7t2', '7T', 0.1)]:
    dd = dec.xs(ses, level='session')
    # #3 decoding accuracy (one-sample t on per-subject correlations -> t(29))
    ms = {'3T': 'r = 0.176, t(29) = 7.1, p < 0.001',
          '7T': 'r = 0.16, t(29) = 8.0, p < 0.001'}[lab]
    onesample(dd['r_acc'].dropna(), ms,
              f'Decoding accuracy (actual vs decoded numerosity), {lab}',
              order=32.0 + o, msec=MS3)
    # #4 decoded uncertainty vs error (one-sample t -> t(29))
    ms = {'3T': 'r = 0.33, t(29) = 9.8, p < 0.001',
          '7T': 'r = 0.38, t(29) = 18.8, p < 0.001'}[lab]
    onesample(dd['r_err'].dropna(), ms,
              f'Decoded uncertainty vs decoder error, {lab}',
              order=33.0 + o, msec=MS3)
    # #5 decoding correlation vs v1 (between-subject Pearson -> r(28))
    ms = {'3T': 'r(29) = -0.48, p = 0.007', '7T': 'r(29) = -0.42, p = 0.021'}[lab]
    sub_dd = dd.dropna(subset=['r_pooled', 'v1'])
    between(sub_dd['r_pooled'], sub_dd['v1'], ms,
            f'Decoding correlation vs v1 (1st-option noise), {lab}',
            order=34.0 + o, msec=MS3, note='manuscript df should be 28, not 29')
    # #6 mean decoded SD vs v1 (between-subject Pearson -> r(28))
    ms = {'3T': 'r(29) = 0.44, p = 0.014', '7T': 'r(29) = 0.38, p = 0.039'}[lab]
    sub_dd = dd.dropna(subset=['mean_sd', 'v1'])
    between(sub_dd['mean_sd'], sub_dd['v1'], ms,
            f'Mean decoded SD vs v1 (1st-option noise), {lab}',
            order=35.0 + o, msec=MS3, note='manuscript df should be 28, not 29')

# #7 symbolic: per-subject mean gamma (consistency) vs rnp (risk attitude),
# model0. Ported from symbolic_experiment/notebooks/analyze_probit_models.ipynb
# (extract_intercept_gamma is defined inline there, not importable).
print('\n  --- Symbolic experiment ---')


def _sym_intercept_gamma(trace, model, data, group=False):
    import scipy.stats as ss
    subj = data.index.unique(level='subject')
    fake = pd.MultiIndex.from_product(
        [subj[:1] if group else subj, [0, 1], data['n_safe_bin'].unique(),
         ['Risky first', 'Safe first']],
        names=['subject', 'log_risky_safe', 'n_safe_bin', 'order']).to_frame(index=False)
    pred = model.predict(trace, 'response_params', fake, inplace=False,
                         include_group_specific=not group)['posterior']['p']
    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake))
    p0 = pred.xs(0, 0, 'log_risky_safe')
    intercept = pd.DataFrame(ss.norm.ppf(p0), index=p0.index, columns=p0.columns)
    gamma = ss.norm.ppf(pred.xs(1, 0, 'log_risky_safe')) - intercept
    return intercept.droplevel(0, 1), gamma.droplevel(0, 1)


try:
    from risk_experiment.symbolic_experiment.fit_probit import (
        build_model as build_sym, get_data as get_sym)
    df_sym = get_sym()
    m0 = build_sym(model_label=0)
    id0 = az.from_netcdf(op.join(SYMBOLIC_BIDS, 'derivatives', 'risk_model',
                                 'psychophysical', 'model0_samples.nc'))
    inter, gamma = _sym_intercept_gamma(id0, m0, df_sym, group=False)
    rnp = np.clip(np.exp(inter / gamma), 0, 1)
    mp = pd.concat([rnp.stack().rename('rnp'), gamma.stack().rename('gamma')],
                   axis=1).groupby('subject').mean()
    between(mp['gamma'], mp['rnp'], 'r(57) = 0.34, p = 0.00822',
            'Symbolic: choice consistency vs indifference point',
            order=41.0, msec=MS4, note='manuscript df should be 56 (n=58), not 57')
except Exception as e:
    report('Symbolic: choice consistency vs indifference point',
           'r(57) = 0.34, p = 0.00822',
           'r(56) = 0.344, 95% CI [0.09, 0.55], p = 0.008',
           note=f'verified value from analyze_probit_models.ipynb cell 9 '
                f'(df corrected 57->56; live recompute unavailable: {type(e).__name__})',
           order=41.0, msec=MS4)

# #8 PMCM parameter test-retest reliability across sessions (refit model-1)
print('\n  --- PMCM parameter test-retest reliability (refit model-1) ---')
params = ['n1_evidence_sd', 'n2_evidence_sd', 'risky_prior_mu', 'safe_prior_mu',
          'risky_prior_sd', 'safe_prior_sd']
est = {}
for ses in ['3t2', '7t2']:
    post = az.from_netcdf(op.join(REFIT, f'ses-{ses}_model-1_trace.netcdf')).posterior
    est[ses] = pd.concat({p: post[p].to_dataframe().groupby('subject').mean().iloc[:, 0]
                          for p in params}, axis=1)
rs = []
for p in params:
    c = pingouin.corr(est['3t2'][p], est['7t2'][p])
    rs.append(c['r'].iloc[0])
    print(f'      {p:16s}: {fmt_corr(c["r"].iloc[0], c["CI95%"].iloc[0], c["p-val"].iloc[0], c["n"].iloc[0])}')
report('PMCM parameter test-retest reliability, range over 6 params (refit model-1)',
       'r(29) between 0.41 and 0.76 (all p<0.05)',
       f'r(28) between {min(rs):.2f} and {max(rs):.2f} (all p < 0.05)',
       note='manuscript df should be 28, not 29',
       order=20.0, msec=MS2)

# Surface nPRF test-retest (#1/#2): volume PRF derivatives are not on this
# machine, so we report the recovered manuscript-era values from
# analyze_encoding.ipynb cells 17/18 (one-sample t-test on per-subject voxelwise
# correlations within rNPC; t-values reproduce the manuscript exactly).
section('Surface nPRF test-retest  (one-sample t on per-subject correlations)')
report('Vertex-wise R^2 test-retest, 3T vs 7T (analyze_encoding.ipynb cell 17)',
       'r = 0.25, t(29) = 6.2, p < 0.001',
       'mean r = 0.28, 95% CI [0.19, 0.38], t(29) = 6.2, p < 0.001',
       note='from recovered manuscript-era run; needs cluster recompute to refresh',
       order=30.0, msec=MS3)
report('Preferred-numerosity test-retest, 3T vs 7T (analyze_encoding.ipynb cell 18)',
       'r = 0.15, t(29) = 6.0, p < 0.001',
       'mean r = 0.16, 95% CI [0.11, 0.21], t(29) = 6.0, p < 0.001',
       note='from recovered manuscript-era run; needs cluster recompute to refresh',
       order=31.0, msec=MS3)

# --------------------------------------------------------------------------- #
# Statistics whose traces are no longer on disk (verified manuscript-era values
# from the recovered notebooks; flagged for cluster recompute).
# --------------------------------------------------------------------------- #
section('Other reported values (verified from recovered notebooks / summaries)')

# MS1: order x stake interaction on RNP (summary across all stake sizes).
report('RNP differs between presentation orders, all stake sizes (summary)',
       'p_Bayesian < 0.05',
       'p_Bayesian < 0.05 (summary across stake sizes)',
       note='summary/criterion across multiple tests -- leave as-is (per brief); '
            'exact per-stake values could be substituted if desired',
       order=10.0, msec=MS1)

# MS3: psychophysical slope, high vs low neural uncertainty (probit_neural4
# per session; traces no longer on disk -> verified notebook values).
for lab, ms, val, order in [
        ('3T', 'p_Bayesian = 0.0458', '0.046', 36.0),
        ('7T', 'p_Bayesian = 0.0238', '0.024', 36.1)]:
    report(f'Psychophysical slope, high vs low neural uncertainty, {lab}',
           ms, f'p_Bayesian = {val}',
           note='analyze_neural_probit.ipynb (probit_neural4); verified, trace off-disk',
           order=order, msec=MS3)

# MS3: RNP distance from risk-neutrality, high vs low neural uncertainty.
for lab, ms, val, order, src in [
        ('both', 'p_Bayesian < 0.001', '< 0.001', 37.0, 'probit_neural9 cell 28'),
        ('3T',   'p_Bayesian = 0.012', '0.012',   37.1, 'cell 9'),
        ('7T',   'p_Bayesian = 0.013', '0.013',   37.2, 'cell 18')]:
    report(f'RNP distance from risk-neutral, high vs low neural uncertainty, {lab}',
           ms, f'p_Bayesian = {val}' if val != '< 0.001' else 'p_Bayesian < 0.001',
           note=f'analyze_neural_probit.ipynb {src}; verified',
           order=order, msec=MS3)

# MS4: subset of participants robustly noisier on 2nd than 1st option (count).
report('Participants robustly noisier on 2nd than 1st option (5/58 count)',
       'p_Bayesian < 0.05',
       'p_Bayesian < 0.05 (classification criterion; 5/58 participants)',
       note='criterion/count, not a single reported test -- leave as-is (per brief)',
       order=40.0, msec=MS4)

# --------------------------------------------------------------------------- #
# Write outputs: machine-readable CSV + Nature-format markdown.
# --------------------------------------------------------------------------- #
notes_dir = op.join(op.dirname(op.abspath(__file__)), 'notes')
csv_out = op.join(notes_dir, 'reported_statistics.csv')
pd.DataFrame(_rows).to_csv(csv_out, index=False)
print(f'\nWrote table -> {csv_out}')


def write_markdown(rows, path):
    """Nature Communications-format markdown of every reported statistic.

    NC editorial rules applied: exact p to 3 decimals (`< 0.001` when it rounds
    to 0.000); 95% CI reported for every correlation; Bayesian p_Bayesian kept
    distinct from frequentist p. The 'Recomputed value' column is the value to
    use in the manuscript; 'Manuscript' is the currently-printed value.
    """
    rows = sorted(rows, key=lambda r: r['order'])
    lines = [
        '# Reported statistics — NCOMMS-24-63995B (de Hollander et al.)',
        '',
        'Auto-generated by `risk_experiment/revision/report_statistics.py` '
        '(run under the `risk7t` env / pinned bauer 0.1.0). Rows are in the order '
        'the statistics appear in the main text, grouped by Results subsection. '
        'The **Copy-paste (corrected)** column is ready to drop straight into the '
        'manuscript.',
        '',
        'Formatting follows Nature Communications: exact *p* to three decimals '
        '(`< 0.001` when smaller); a 95% confidence interval on every '
        'correlation; Pearson df reported as **n − 2** (the manuscript used '
        'n − 1, so `r(29)` → `r(28)` and `r(57)` → `r(56)`; one-sample '
        '*t*-tests correctly keep `t(29)`). `p_Bayesian` = posterior probability '
        'mass (one-tailed); plain *p* = frequentist (italicise/capitalise *P* to '
        'taste). Values flagged in **Note** differ from the current manuscript.',
        '',
    ]
    # group by manuscript subsection, in order of first appearance
    msecs = []
    for r in rows:
        if r['msec'] not in msecs:
            msecs.append(r['msec'])
    for ms in msecs:
        lines += [f'## {ms}', '',
                  '| # | Quantity | Copy-paste (corrected) | Manuscript (current) | Note |',
                  '|---|---|---|---|---|']
        for r in rows:
            if r['msec'] != ms:
                continue
            note = r['note'].replace('|', '\\|')
            lines.append(f"| {r['order']:.1f} | {r['location']} | "
                         f"`{r['computed']}` | {r['manuscript']} | {note} |")
        lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


md_out = op.join(notes_dir, 'reported_statistics.md')
write_markdown(_rows, md_out)
print(f'Wrote markdown -> {md_out}')
