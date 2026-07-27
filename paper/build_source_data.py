#!/usr/bin/env python3
"""Build Source_Data.xlsx for de Hollander et al. (Nature Communications).

Unlike the older ``create_source_data.py`` (which re-derived every quantity and
used the pre-revision figure numbering), this builder reads the *exact* per-panel
source-data TSVs that the figure scripts emit into
``risk_experiment/revision/figures/source_data/``. Those TSVs are what the panels
plot, so the workbook is guaranteed consistent with the figures. Sample-based
panels (KDEs / forests over posterior draws) are summarised to the plotted
quantities (mean, 95% HDI, etc.); curves/points are written as-is.

New 6-figure numbering (post-revision):
  1 behaviour · 2 PMCM · 3 alt-model PPCs · 4 ELPD · 5 neural · 6 symbolic

Run from the repo root:  python paper/build_source_data.py
Output: paper/Source_Data.xlsx
"""
import os.path as op

import arviz as az
import numpy as np
import pandas as pd

ROOT = op.dirname(op.dirname(op.abspath(__file__)))
SRC = op.join(ROOT, 'risk_experiment', 'revision', 'figures', 'source_data')
OUTPUT = op.join(op.dirname(op.abspath(__file__)), 'Source_Data.xlsx')

ORDER = {True: 'Risky first', False: 'Safe first'}


def tsv(name):
    return pd.read_csv(op.join(SRC, f'{name}.tsv'), sep='\t')


def hdi_lo(x):
    return az.hdi(np.asarray(x, float), hdi_prob=0.95)[0]


def hdi_hi(x):
    return az.hdi(np.asarray(x, float), hdi_prob=0.95)[1]


sheets = {}

# --- Figure 1b: prop. risky vs safe offer x order -------------------------
b = tsv('figure_01_mag_order_effect').rename(
    columns={'n_safe': 'safe_payoff_CHF', 'chose_risky': 'prop_chose_risky'})
sheets['Figure 1b (per subject)'] = b[['subject', 'safe_payoff_CHF', 'Order',
                                       'prop_chose_risky']]
sheets['Figure 1b'] = (b.groupby(['safe_payoff_CHF', 'Order'])['prop_chose_risky']
                       .agg(mean='mean', sem='sem').reset_index())

# --- Figure 1c: psychophysical curves + RNP -------------------------------
c = tsv('figure_01_rnp_stake_effect').rename(
    columns={'bin(stake size)': 'stake_bin', 'bin(risky/safe)': 'ratio_bin',
             'chose_risky': 'prop_chose_risky', 'logrs': 'mean_log_risky_safe'})
sheets['Figure 1c (per subject)'] = c[['subject', 'stake_bin', 'Order',
                                       'ratio_bin', 'prop_chose_risky',
                                       'mean_log_risky_safe']]
sheets['Figure 1c'] = (c.groupby(['stake_bin', 'Order', 'ratio_bin'])
                       .agg(prop_chose_risky_mean=('prop_chose_risky', 'mean'),
                            prop_chose_risky_sem=('prop_chose_risky', 'sem'),
                            mean_log_risky_safe=('mean_log_risky_safe', 'mean'))
                       .reset_index())

rnp = tsv('figure_01_rnp_stake_effect_rnp')
sheets['Figure 1c RNP'] = (rnp.groupby(['n_safe_bin', 'Order'])['rnp']
                           .agg(rnp_mean='mean', rnp_hdi_2_5=hdi_lo,
                                rnp_hdi_97_5=hdi_hi).reset_index()
                           .rename(columns={'n_safe_bin': 'stake_bin'}))

# --- Figure 2a: PMCM posterior-predictive ---------------------------------
a = tsv('figure_02A_ppc').rename(columns={
    'bin(n_safe)': 'stake_bin', 'bin(risky/safe)': 'ratio_bin',
    'chose_risky': 'empirical_mean', 'pred_mean': 'model_mean',
    'hdi_lower': 'model_hdi_2_5', 'hdi_upper': 'model_hdi_97_5'})
a['Order'] = a['risky_first'].map(ORDER)
sheets['Figure 2a'] = a[['stake_bin', 'Order', 'ratio_bin', 'empirical_mean',
                         'model_mean', 'model_hdi_2_5', 'model_hdi_97_5']]

# --- Figure 2b: group-level posteriors (summarised KDEs) ------------------
gp = tsv('figure_02B_group_posteriors')
sheets['Figure 2b'] = (gp.groupby(['panel', 'condition'])['value']
                       .agg(posterior_mean='mean', posterior_sd='std',
                            hdi_2_5=hdi_lo, hdi_97_5=hdi_hi).reset_index()
                       .rename(columns={'panel': 'parameter'}))

# --- Figure 2c: participant-level differences -----------------------------
pc = tsv('figure_02C_participant_diffs').rename(
    columns={'mean': 'posterior_mean', 'low': 'hdi_2_5', 'high': 'hdi_97_5',
             'panel': 'comparison'})
sheets['Figure 2c'] = pc[['subject', 'comparison', 'posterior_mean', 'hdi_2_5',
                          'hdi_97_5', 'risk_profile']]

# --- Figure 3: alternative-model PPCs (A-D) -------------------------------
MODELS = {'modelA_klw': 'Model A: shared prior, equal noise',
          'modelB_52': 'Model B: varying priors, equal noise',
          'modelC_42': 'Model C: shared prior, varying noise',
          'modelD_eu': 'Model D: expected utility'}
f3 = []
for key, name in MODELS.items():
    m = tsv(f'figure_03_{key}').rename(columns={
        'bin(n_safe)': 'stake_bin', 'bin(risky/safe)': 'ratio_bin',
        'chose_risky': 'empirical_mean', 'pred_mean': 'model_mean',
        'hdi_lower': 'model_hdi_2_5', 'hdi_upper': 'model_hdi_97_5'})
    m['Order'] = m['risky_first'].map(ORDER)
    m['model'] = name
    f3.append(m[['model', 'stake_bin', 'Order', 'ratio_bin', 'empirical_mean',
                 'model_mean', 'model_hdi_2_5', 'model_hdi_97_5']])
sheets['Figure 3'] = pd.concat(f3, ignore_index=True)

# --- Figure 4: ELPD model comparison --------------------------------------
elpd = tsv('figure_04_model_comparison').rename(columns={'Unnamed: 0': 'model'})
sheets['Figure 4'] = elpd.drop(columns=['warning'], errors='ignore')

# --- Figure 5b: decoding correlations -------------------------------------
dec = tsv('figure_05_decoding').rename(columns={
    'r_E_n1': 'r_decoded_mean_vs_log_numerosity',
    'r_sd_error': 'r_decoded_sd_vs_abs_error'})
sheets['Figure 5b'] = dec[['subject', 'session', 'Scanner',
                           'r_decoded_mean_vs_log_numerosity',
                           'r_decoded_sd_vs_abs_error']]

# --- Figure 5c: distance to risk-neutral, low vs high neural uncertainty --
dist = tsv('figure_05_rnp_distance')
sheets['Figure 5c'] = (dist.groupby('Neural uncertainty')['distance_to_risk_neutral']
                       .agg(mean_distance='mean', hdi_2_5=hdi_lo, hdi_97_5=hdi_hi)
                       .reset_index())

# --- Figure 5d: neural-uncertainty regressor on PMCM parameters -----------
gam = tsv('figure_05_neural_gamma')
sheets['Figure 5d'] = (gam.groupby(['Parameter', 'Option'])['value']
                       .agg(slope_mean='mean', slope_sd='std', hdi_2_5=hdi_lo,
                            hdi_97_5=hdi_hi,
                            p_slope_negative=lambda x: float((x < 0).mean()))
                       .reset_index())

# --- Figure 6b: symbolic psychophysical curves (empirical + probit PPC) ---
sc = tsv('figure_06_symbolic_psychophysics_curves')
emp = (sc.groupby(['n_safe_bin', 'Order', 'bin(risky/safe)'])['chose_risky']
       .first().rename('empirical_prop_chose_risky'))
pred = (sc.groupby(['n_safe_bin', 'Order', 'bin(risky/safe)'])['chose_risky_pred']
        .agg(model_mean='mean', model_hdi_2_5=hdi_lo, model_hdi_97_5=hdi_hi))
sheets['Figure 6b'] = (pd.concat([emp, pred], axis=1).reset_index()
                       .rename(columns={'n_safe_bin': 'stake_bin',
                                        'bin(risky/safe)': 'log_risky_safe_bin'}))

# --- Figure 6c/6d: symbolic RNP per order + the order difference ----------
srnp = tsv('figure_06_symbolic_psychophysics_rnp')
sheets['Figure 6c (RNP)'] = (srnp.groupby(['n_safe_bin', 'order'])['rnp']
                             .agg(rnp_mean='mean', rnp_hdi_2_5=hdi_lo,
                                  rnp_hdi_97_5=hdi_hi).reset_index()
                             .rename(columns={'n_safe_bin': 'stake_bin'}))
wide = srnp.set_index(['n_safe_bin', 'order', 'chain', 'draw'])['rnp'].unstack('order')
delta = (wide['Safe first'] - wide['Risky first']).rename('rnp_difference')
sheets['Figure 6d (delta RNP)'] = (delta.groupby('n_safe_bin')
                                   .agg(delta_mean='mean', hdi_2_5=hdi_lo,
                                        hdi_97_5=hdi_hi).reset_index()
                                   .rename(columns={'n_safe_bin': 'stake_bin'}))

# --- write workbook -------------------------------------------------------
with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
    for name, data in sheets.items():
        data.to_excel(writer, sheet_name=name[:31], index=False)
        print(f'  {name[:31]:34s} {data.shape}')
print(f'\nWrote {OUTPUT}')
