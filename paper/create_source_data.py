#!/usr/bin/env python3
"""
Create 'Source Data' Excel file for de Hollander et al. 2024
"Rapid Changes in Risk Attitudes Originate from Bayesian Inference on Parietal Magnitude Representations"

Each sheet contains data underlying one figure panel.
Run from the repo root: python paper/create_source_data.py

Output: paper/Source_Data.xlsx
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import arviz as az
import os.path as op
import pingouin
from itertools import product as iproduct
from bauer.utils.math import softplus_np

BIDS_FOLDER = '/data/ds-risk'
SYMBOLIC_BIDS_FOLDER = '/data/ds-symbolicrisk'
OUTPUT = op.join(op.dirname(op.abspath(__file__)), 'Source_Data.xlsx')

print("Loading behavioral data...")
from risk_experiment.utils import get_all_behavior, get_all_subjects, get_all_subject_ids
df = get_all_behavior(bids_folder=BIDS_FOLDER)

sheets = {}

# ---------------------------------------------------------------------------
# FIGURE 1B
# Average proportion of risky choices as a function of safe payoff and order
# ---------------------------------------------------------------------------
print("Extracting Figure 1B data...")

fig1b_sub = (
    df.groupby(['subject', 'n_safe', 'risky_first'])['chose_risky']
    .mean()
    .reset_index()
)
fig1b_sub.columns = ['subject', 'safe_payoff_CHF', 'risky_first', 'prop_chose_risky']
fig1b_sub['order'] = fig1b_sub['risky_first'].map({True: 'Risky first', False: 'Safe first'})

fig1b = (
    fig1b_sub.groupby(['safe_payoff_CHF', 'order'])['prop_chose_risky']
    .agg(mean='mean', sem='sem')
    .reset_index()
)

sheets['Figure 1B'] = fig1b
sheets['Figure 1B (per subject)'] = fig1b_sub[['subject', 'safe_payoff_CHF', 'order', 'prop_chose_risky']]

# ---------------------------------------------------------------------------
# FIGURE 1C
# Psychophysical curves: proportion risky choice vs risky/safe ratio
# binned by stake size and order
# ---------------------------------------------------------------------------
print("Extracting Figure 1C data...")

df['stake_bin'] = pd.qcut(df['n_safe'], q=3, labels=['Small (5-7)', 'Medium (10-14)', 'Large (20-28)'])

fig1c_sub = (
    df.groupby(['subject', 'stake_bin', 'risky_first', 'bin(risky/safe)'])['chose_risky']
    .mean()
    .reset_index()
)
fig1c_sub.columns = ['subject', 'stake_bin', 'risky_first', 'log_risky_safe_ratio_bin', 'prop_chose_risky']
fig1c_sub['order'] = fig1c_sub['risky_first'].map({True: 'Risky first', False: 'Safe first'})

fig1c = (
    fig1c_sub.groupby(['stake_bin', 'order', 'log_risky_safe_ratio_bin'])['prop_chose_risky']
    .agg(mean='mean', sem='sem')
    .reset_index()
)

sheets['Figure 1C'] = fig1c
sheets['Figure 1C (per subject)'] = fig1c_sub[['subject', 'stake_bin', 'order', 'log_risky_safe_ratio_bin', 'prop_chose_risky']]

# RNP insets: extract from probit model (probit_full_session)
print("  Loading probit model for Figure 1C RNP insets...")
try:
    from risk_experiment.cogmodels.fit_probit import build_model as build_probit, get_data as get_probit_data
    from risk_experiment.cogmodels.utils import extract_intercept_gamma

    # Combine both sessions
    df_probit = get_probit_data(model_label='probit_full', session=None, bids_folder=BIDS_FOLDER)
    model_probit = build_probit(model_label='probit_full', df=df_probit, session=None, bids_folder=BIDS_FOLDER)
    idata_probit = az.from_netcdf(op.join(BIDS_FOLDER, 'derivatives', 'cogmodels', 'model-probit_full_session_trace.netcdf'))

    # Group-level RNP
    intercept, gamma = extract_intercept_gamma(idata_probit, model_probit, df_probit, group=True)
    rnp_raw = np.exp(intercept['intercept'] / gamma['gamma'])

    rnp_df = rnp_raw.stack([-2, -1]).to_frame('rnp')
    rnp_df = rnp_df.reset_index()
    rnp_df['order'] = rnp_df['risky_first'].map({True: 'Risky first', False: 'Safe first'})
    rnp_df['stake_bin'] = pd.qcut(rnp_df['n_safe'], q=3, labels=['Small (5-7)', 'Medium (10-14)', 'Large (20-28)'])

    rnp_summary = (
        rnp_df.groupby(['stake_bin', 'order'])['rnp']
        .agg(mean='mean',
             hdi_2_5=lambda x: az.hdi(x.values, hdi_prob=0.95)[0],
             hdi_97_5=lambda x: az.hdi(x.values, hdi_prob=0.95)[1])
        .reset_index()
    )
    sheets['Figure 1C RNP insets'] = rnp_summary
except Exception as e:
    print(f"  Warning: could not extract RNP insets: {e}")

# ---------------------------------------------------------------------------
# FIGURE 2A
# PMC model posterior predictive check: empirical data + model predictions
# ---------------------------------------------------------------------------
print("Extracting Figure 2A data (PMC model PPC)...")

try:
    from risk_experiment.cogmodels.fit_model import build_model as build_pmcm, get_data as get_pmcm_data

    model_label = '12'
    df_pmcm = get_pmcm_data(model_label=model_label, session=None, bids_folder=BIDS_FOLDER, roi=None)
    # Add stake_bin grouping to model dataframe
    df_pmcm['stake_bin'] = pd.qcut(df_pmcm['n_safe'], q=3, labels=['Small (5-7)', 'Medium (10-14)', 'Large (20-28)'])
    model_pmcm = build_pmcm(df=df_pmcm, model_label=model_label, roi=None)
    idata_pmcm = az.from_netcdf(op.join(BIDS_FOLDER, 'derivatives', 'cogmodels', f'model-{model_label}_trace.netcdf'))

    # PPC: subsample every 20th draw to keep memory manageable
    idata_sub = idata_pmcm.sel(draw=slice(None, None, 20))
    ppc = model_pmcm.ppc(
        idata=idata_sub,
        paradigm=df_pmcm.drop('session', axis=1),
        var_names=['ll_bernoulli']
    )

    ppc_df = ppc.stack([0, 1]).to_frame('choice_pred')
    ppc_df['choice_pred'] = ppc_df['choice_pred'].astype(float)
    ppc_df = ppc_df.reset_index().set_index(df_pmcm.index.names + ['chain', 'draw'])[['choice_pred']]
    ppc_df = ppc_df.join(df_pmcm[['risky_first', 'n_risky', 'n_safe', 'chose_risky', 'bin(risky/safe)', 'stake_bin']])
    ppc_df['chose_risky_pred'] = ppc_df.apply(
        lambda r: 1 - r['choice_pred'] if r['risky_first'] else r['choice_pred'], axis=1
    )

    ppc_summary = (
        ppc_df.groupby(['stake_bin', 'risky_first', 'bin(risky/safe)', 'chain', 'draw'])['chose_risky_pred']
        .mean()
        .groupby(['stake_bin', 'risky_first', 'bin(risky/safe)'])
        .agg(model_mean='mean',
             model_hdi_2_5=lambda x: az.hdi(x.values, hdi_prob=0.95)[0],
             model_hdi_97_5=lambda x: az.hdi(x.values, hdi_prob=0.95)[1])
        .reset_index()
    )

    emp = (
        df_pmcm.groupby(['stake_bin', 'risky_first', 'bin(risky/safe)'])['chose_risky']
        .agg(empirical_mean='mean', empirical_sem='sem')
        .reset_index()
    )

    fig2a = ppc_summary.merge(emp, on=['stake_bin', 'risky_first', 'bin(risky/safe)'])
    fig2a['order'] = fig2a['risky_first'].map({True: 'Risky first', False: 'Safe first'})
    sheets['Figure 2A'] = fig2a

except Exception as e:
    print(f"  Warning: could not generate Figure 2A PPC: {e}")
    # Fall back to empirical data only
    emp = (
        df.groupby(['stake_bin', 'risky_first', 'bin(risky/safe)'])['chose_risky']
        .agg(empirical_mean='mean', empirical_sem='sem')
        .reset_index()
    )
    emp['order'] = emp['risky_first'].map({True: 'Risky first', False: 'Safe first'})
    sheets['Figure 2A (empirical only)'] = emp

# ---------------------------------------------------------------------------
# FIGURE 2B
# Group-level parameter posteriors from PMC model (both sessions combined)
# Uses model-12 with sessions averaged: softplus(Intercept + 0.5*session[T.7t2])
# ---------------------------------------------------------------------------
print("Extracting Figure 2B data (PMC model group parameters, combined sessions)...")

idata_12 = az.from_netcdf(op.join(BIDS_FOLDER, 'derivatives', 'cogmodels', 'model-12_trace.netcdf'))

def combined_session(idata, param):
    """Average 3T and 7T: softplus(Intercept + 0.5 * session[T.7t2])"""
    df_param = idata.posterior[param].to_dataframe()
    intercept = df_param.xs('Intercept', level=-1)
    session_effect = df_param.xs('session[T.7t2]', level=-1)
    return softplus_np(intercept.iloc[:, 0] + 0.5 * session_effect.iloc[:, 0])

fig2b_rows = []
for param, label in [
    ('n1_evidence_sd_mu', 'n1_evidence_sd (1st option noise)'),
    ('n2_evidence_sd_mu', 'n2_evidence_sd (2nd option noise)'),
    ('risky_prior_mu_mu', 'risky_prior_mu (risky prior mean)'),
    ('safe_prior_mu_mu',  'safe_prior_mu (safe prior mean)'),
    ('risky_prior_sd_mu', 'risky_prior_sd (risky prior width)'),
    ('safe_prior_sd_mu',  'safe_prior_sd (safe prior width)'),
]:
    vals = combined_session(idata_12, param).values
    hdi95 = az.hdi(vals, hdi_prob=0.95)
    fig2b_rows.append({
        'parameter': label,
        'posterior_mean': float(vals.mean()),
        'posterior_sd': float(vals.std()),
        'hdi_2_5': float(hdi95[0]),
        'hdi_97_5': float(hdi95[1]),
    })

sheets['Figure 2B'] = pd.DataFrame(fig2b_rows)

# ---------------------------------------------------------------------------
# FIGURE 2C
# Subject-level parameter differences (combined sessions via model-12)
# softplus(Intercept + 0.5 * session[T.7t2]) per subject
# ---------------------------------------------------------------------------
print("Extracting Figure 2C data (subject-level parameter differences, combined sessions)...")

def combined_session_subject(idata, param):
    """Per-subject session average: softplus(Intercept + 0.5 * session[T.7t2])"""
    df_param = idata.posterior[param].to_dataframe()
    intercept = df_param.xs('Intercept', level=-1)
    session_effect = df_param.xs('session[T.7t2]', level=-1)
    return softplus_np(intercept.iloc[:, 0] + 0.5 * session_effect.iloc[:, 0])

risk_prefs = pd.read_csv(
    op.join(BIDS_FOLDER, 'derivatives', 'cogmodels', 'simple_risk_preference.tsv'),
    index_col=[0, 1], sep='\t', dtype={'subject': str}
)

fig2c_rows = []

# n1 - n2 evidence_sd
n1_sd = combined_session_subject(idata_12, 'n1_evidence_sd')
n2_sd = combined_session_subject(idata_12, 'n2_evidence_sd')
diff_sd = (n1_sd - n2_sd).groupby('subject')
for sub, grp in diff_sd:
    hdi95 = az.hdi(grp.values, hdi_prob=0.95)
    fig2c_rows.append({
        'subject': sub,
        'comparison': 'noise_n1_minus_n2',
        'description': 'Evidence SD: 1st option - 2nd option',
        'posterior_mean': float(grp.mean()),
        'hdi_2_5': float(hdi95[0]),
        'hdi_97_5': float(hdi95[1]),
        'p_positive': float((grp.values > 0).mean()),
    })

# risky - safe prior_mu
r_mu = combined_session_subject(idata_12, 'risky_prior_mu')
s_mu = combined_session_subject(idata_12, 'safe_prior_mu')
diff_mu = (r_mu - s_mu).groupby('subject')
for sub, grp in diff_mu:
    hdi95 = az.hdi(grp.values, hdi_prob=0.95)
    fig2c_rows.append({
        'subject': sub,
        'comparison': 'prior_mu_risky_minus_safe',
        'description': 'Prior mean: risky - safe',
        'posterior_mean': float(grp.mean()),
        'hdi_2_5': float(hdi95[0]),
        'hdi_97_5': float(hdi95[1]),
        'p_positive': float((grp.values > 0).mean()),
    })

# risky - safe prior_sd
r_sd = combined_session_subject(idata_12, 'risky_prior_sd')
s_sd = combined_session_subject(idata_12, 'safe_prior_sd')
diff_prsd = (r_sd - s_sd).groupby('subject')
for sub, grp in diff_prsd:
    hdi95 = az.hdi(grp.values, hdi_prob=0.95)
    fig2c_rows.append({
        'subject': sub,
        'comparison': 'prior_sd_risky_minus_safe',
        'description': 'Prior width: risky - safe',
        'posterior_mean': float(grp.mean()),
        'hdi_2_5': float(hdi95[0]),
        'hdi_97_5': float(hdi95[1]),
        'p_positive': float((grp.values > 0).mean()),
    })

sheets['Figure 2C'] = pd.DataFrame(fig2c_rows)

# ---------------------------------------------------------------------------
# FIGURE 3A-D
# Posterior predictive checks for alternative models
# (empirical data shared; model PPCs generated per model)
# ---------------------------------------------------------------------------
print("Extracting Figure 3 data (alternative models)...")

ALT_MODELS = {
    'klw': 'Model A: Shared prior, equal noise',
    '52':  'Model B: Varying priors, equal noise',
    '42':  'Model C: Shared prior, varying noise',
    'eu':  'Model D: Expected utility',
}

fig3_rows = []
for model_label, model_name in ALT_MODELS.items():
    print(f"  Processing {model_name} ({model_label})...")
    try:
        df_alt = get_pmcm_data(model_label=model_label, session=None, bids_folder=BIDS_FOLDER, roi=None)
        df_alt['stake_bin'] = pd.qcut(df_alt['n_safe'], q=3, labels=['Small (5-7)', 'Medium (10-14)', 'Large (20-28)'])
        model_alt = build_pmcm(df=df_alt, model_label=model_label, roi=None)
        idata_alt = az.from_netcdf(op.join(BIDS_FOLDER, 'derivatives', 'cogmodels', f'model-{model_label}_trace.netcdf'))

        idata_sub = idata_alt.sel(draw=slice(None, None, 20))
        ppc_alt = model_alt.ppc(
            idata=idata_sub,
            paradigm=df_alt.drop('session', axis=1),
            var_names=['ll_bernoulli']
        )
        ppc_alt_df = ppc_alt.stack([0, 1]).to_frame('choice_pred')
        ppc_alt_df['choice_pred'] = ppc_alt_df['choice_pred'].astype(float)
        ppc_alt_df = ppc_alt_df.reset_index().set_index(df_alt.index.names + ['chain', 'draw'])[['choice_pred']]
        ppc_alt_df = ppc_alt_df.join(df_alt[['risky_first', 'n_safe', 'chose_risky', 'bin(risky/safe)', 'stake_bin']])
        ppc_alt_df['chose_risky_pred'] = ppc_alt_df.apply(
            lambda r: 1 - r['choice_pred'] if r['risky_first'] else r['choice_pred'], axis=1
        )
        ppc_alt_sum = (
            ppc_alt_df.groupby(['stake_bin', 'risky_first', 'bin(risky/safe)', 'chain', 'draw'])['chose_risky_pred']
            .mean()
            .groupby(['stake_bin', 'risky_first', 'bin(risky/safe)'])
            .agg(model_mean='mean',
                 model_hdi_2_5=lambda x: az.hdi(x.values, hdi_prob=0.95)[0],
                 model_hdi_97_5=lambda x: az.hdi(x.values, hdi_prob=0.95)[1])
            .reset_index()
        )
        emp_alt = (
            df_alt.groupby(['stake_bin', 'risky_first', 'bin(risky/safe)'])['chose_risky']
            .agg(empirical_mean='mean', empirical_sem='sem')
            .reset_index()
        )
        merged = ppc_alt_sum.merge(emp_alt, on=['stake_bin', 'risky_first', 'bin(risky/safe)'])
        merged['model'] = model_name
        merged['order'] = merged['risky_first'].map({True: 'Risky first', False: 'Safe first'})
        fig3_rows.append(merged)
    except Exception as e:
        print(f"  Warning: could not run PPC for {model_name}: {e}")

if fig3_rows:
    sheets['Figure 3A-D'] = pd.concat(fig3_rows, ignore_index=True)

# ---------------------------------------------------------------------------
# FIGURE 3E
# Model comparison: ELPD values
# ---------------------------------------------------------------------------
print("Extracting Figure 3E data (ELPD model comparison)...")

try:
    import pymc as pm

    all_models = dict(ALT_MODELS)
    all_models['12'] = 'PMC model'

    idatas_comparison = {}
    for model_label, model_name in all_models.items():
        df_m = get_pmcm_data(model_label=model_label, session=None, bids_folder=BIDS_FOLDER, roi=None)
        model_m = build_pmcm(df=df_m, model_label=model_label, roi=None)
        idata_m = az.from_netcdf(op.join(BIDS_FOLDER, 'derivatives', 'cogmodels', f'model-{model_label}_trace.netcdf'))

        if 'log_likelihood' not in idata_m.groups():
            with model_m.estimation_model:
                ll = pm.compute_log_likelihood(idata_m)
            try:
                idata_m.add_groups({'log_likelihood': ll})
            except Exception:
                pass  # already exists

        idatas_comparison[model_name] = idata_m

    comparison = az.compare(idatas_comparison)
    sheets['Figure 3E'] = comparison.reset_index().rename(columns={'index': 'model'})

except Exception as e:
    print(f"  Warning: could not compute ELPD comparison: {e}")

# Figure 4A is a brain imaging map (preferred numerosity in IPS) and is not
# included as a spreadsheet sheet. The underlying NIfTI files are in
# /data/ds-risk/derivatives/encoding_model.cv.denoise.natural_space/

# ---------------------------------------------------------------------------
# FIGURE 4B
# Decoding accuracy: correlation between decoded and actual numerosity
# ---------------------------------------------------------------------------
print("Extracting Figure 4B data (decoding accuracy)...")

try:
    subjects_obj = get_all_subjects(bids_folder=BIDS_FOLDER)
    fig4b_rows = []
    for sub in subjects_obj:
        for session in ['3t2', '7t2']:
            try:
                pred = sub.get_decoding_info(session, mask='npcr', n_voxels=0)
                behavior = sub.get_behavior(sessions=session)
                merged = pred.join(behavior[['n1']]).dropna(subset=['n1', 'E'])
                if len(merged) > 10:
                    r = pingouin.corr(merged['n1'], merged['E'])['r'].iloc[0]
                    r_sd_error = pingouin.corr(
                        np.abs(merged['n1'] - merged['E']),
                        merged['sd']
                    )['r'].iloc[0]
                    fig4b_rows.append({
                        'subject': sub.subject,
                        'session': session,
                        'scanner': '3T' if session == '3t2' else '7T',
                        'r_decoded_vs_actual_numerosity': r,
                        'r_decoded_uncertainty_vs_abs_error': r_sd_error,
                        'n_trials': len(merged),
                    })
            except Exception:
                pass
    sheets['Figure 4B'] = pd.DataFrame(fig4b_rows)

except Exception as e:
    print(f"  Warning: could not extract decoding data: {e}")

# ---------------------------------------------------------------------------
# FIGURE 4C
# Distance to risk neutrality by decoded neural uncertainty (high vs low)
# ---------------------------------------------------------------------------
print("Extracting Figure 4C data (neural uncertainty and RNP distance)...")

try:
    from risk_experiment.cogmodels.fit_probit import build_model as build_neural_probit, get_data as get_neural_data
    from risk_experiment.cogmodels.utils import extract_intercept_gamma

    # Use combined-session probit_neural9 (session=None)
    df_n = get_neural_data(
        model_label='probit_neural9', session=None, bids_folder=BIDS_FOLDER
    )
    model_n = build_neural_probit(
        model_label='probit_neural9', df=df_n, session=None, bids_folder=BIDS_FOLDER
    )
    idata_n = az.from_netcdf(
        op.join(BIDS_FOLDER, 'derivatives', 'cogmodels', 'model-probit_neural9_trace.netcdf')
    )

    intercept, gamma = extract_intercept_gamma(idata_n, model_n, df_n, group=False)
    rnp = np.exp(intercept['intercept'] / gamma['gamma']).stack([-2, -1]).to_frame('rnp')
    rnp = rnp[(rnp['rnp'] > 0.0) & (rnp['rnp'] < 1.0)]
    rnp['distance_to_risk_neutral'] = (rnp['rnp'] - 0.55).abs()

    fig4c_rows = []
    for neural_unc in [True, False]:
        grp = rnp.xs(neural_unc, level='median_split_sd')['distance_to_risk_neutral']
        vals = grp.groupby(['chain', 'draw']).mean().values
        hdi95 = az.hdi(vals, hdi_prob=0.95)
        fig4c_rows.append({
            'neural_uncertainty': 'High' if neural_unc else 'Low',
            'mean_distance_to_risk_neutral': float(vals.mean()),
            'hdi_2_5': float(hdi95[0]),
            'hdi_97_5': float(hdi95[1]),
        })

    sheets['Figure 4C'] = pd.DataFrame(fig4c_rows)

except Exception as e:
    print(f"  Warning: could not extract Figure 4C data: {e}")

# ---------------------------------------------------------------------------
# FIGURE 4D
# PMCM parameters that co-vary with trial-to-trial decoded neural uncertainty
# ---------------------------------------------------------------------------
print("Extracting Figure 4D data (neural uncertainty effect on model parameters)...")

try:
    idata_neural = az.from_netcdf(
        op.join(BIDS_FOLDER, 'derivatives', 'cogmodels', 'model-neural32_trace.netcdf')
    )

    # neural32 model parameters include a 'sd' regressor that captures
    # how model parameters scale with decoded neural uncertainty (sd)
    fig4d_rows = []
    param_labels = {
        'n1_evidence_sd_mu': '1st option noise (n1_evidence_sd)',
        'n2_evidence_sd_mu': '2nd option noise (n2_evidence_sd)',
        'risky_prior_mu_mu': 'Risky prior mean',
        'safe_prior_mu_mu': 'Safe prior mean',
        'risky_prior_sd_mu': 'Risky prior width',
        'safe_prior_sd_mu': 'Safe prior width',
    }

    for param_key, param_name in param_labels.items():
        if param_key not in idata_neural.posterior:
            continue
        pvar = idata_neural.posterior[param_key]
        coords = dict(pvar.coords)
        # Look for a 'sd' or 'neural_uncertainty' regressor dimension
        regressor_dims = [d for d in pvar.dims if d not in ('chain', 'draw')]
        if regressor_dims:
            dim_name = regressor_dims[0]
            dim_values = list(pvar.coords[dim_name].values)
            if 'sd' in dim_values:
                slope_vals = pvar.sel({dim_name: 'sd'}).values.ravel()
                hdi95 = az.hdi(slope_vals, hdi_prob=0.95)
                p_positive = float((slope_vals > 0).mean())
                fig4d_rows.append({
                    'parameter': param_name,
                    'regressor': 'decoded_neural_uncertainty_sd',
                    'slope_mean': float(slope_vals.mean()),
                    'slope_sd': float(slope_vals.std()),
                    'hdi_2_5': float(hdi95[0]),
                    'hdi_97_5': float(hdi95[1]),
                    'p_positive': p_positive,
                    'p_significant (>0.95 or <0.05)': p_positive > 0.95 or p_positive < 0.05,
                })

    if fig4d_rows:
        sheets['Figure 4D'] = pd.DataFrame(fig4d_rows)

except Exception as e:
    print(f"  Warning: could not extract Figure 4D data: {e}")

# ---------------------------------------------------------------------------
# FIGURE 5B
# Symbolic experiment: psychophysical curves by stake size and order
# ---------------------------------------------------------------------------
print("Extracting Figure 5B data (symbolic experiment)...")

try:
    from risk_experiment.symbolic_experiment.fit_probit import build_model as build_symbolic, get_data as get_symbolic_data

    df_sym = get_symbolic_data()
    model_sym = build_symbolic(model_label=2)
    idata_sym = az.from_netcdf(
        op.join(SYMBOLIC_BIDS_FOLDER, 'derivatives', 'risk_model', 'psychophysical', 'model2_samples.nc')
    )

    # Compute risky/safe ratio bins (as done in the analysis notebooks)
    df_sym['log(risky/safe)'] = np.log(df_sym['n_risky'] / df_sym['n_safe'])
    df_sym['bin(risky/safe)'] = pd.cut(df_sym['log(risky/safe)'], bins=9).apply(lambda x: x.mid)

    # Empirical data
    fig5b_sub = (
        df_sym.groupby(['subject', 'n_safe_bin', 'Order', 'bin(risky/safe)'])['chose_risky']
        .mean()
        .reset_index()
    )
    fig5b_sub.columns = ['subject', 'stake_bin', 'order', 'log_risky_safe_ratio_bin', 'prop_chose_risky']

    fig5b = (
        fig5b_sub.groupby(['stake_bin', 'order', 'log_risky_safe_ratio_bin'])['prop_chose_risky']
        .agg(mean='mean', sem='sem')
        .reset_index()
    )
    sheets['Figure 5B'] = fig5b
    sheets['Figure 5B (per subject)'] = fig5b_sub

    # RNP estimates from symbolic model
    from risk_experiment.symbolic_experiment.fit_probit import extract_intercept_gamma as sym_extract
    intercept_s, gamma_s = sym_extract(idata_sym, model_sym, df_sym, group=True)
    rnp_sym = np.exp(intercept_s['intercept'] / gamma_s['gamma'])

    if hasattr(rnp_sym, 'stack'):
        rnp_sym_df = rnp_sym.stack([-2, -1]).to_frame('rnp') if rnp_sym.ndim > 1 else rnp_sym.to_frame('rnp')
    else:
        rnp_sym_df = rnp_sym.to_frame('rnp')

    rnp_sym_df = rnp_sym_df.reset_index()
    if 'order' in rnp_sym_df.columns or 'Order' in rnp_sym_df.columns:
        order_col = 'order' if 'order' in rnp_sym_df.columns else 'Order'
        stake_col = 'n_safe_bin' if 'n_safe_bin' in rnp_sym_df.columns else 'n_safe'
        rnp_sym_summary = (
            rnp_sym_df.groupby([stake_col, order_col])['rnp']
            .agg(rnp_mean='mean',
                 rnp_hdi_2_5=lambda x: az.hdi(x.values, hdi_prob=0.95)[0],
                 rnp_hdi_97_5=lambda x: az.hdi(x.values, hdi_prob=0.95)[1])
            .reset_index()
        )
        sheets['Figure 5B RNP'] = rnp_sym_summary

except Exception as e:
    print(f"  Warning: could not extract Figure 5B data: {e}")

# ---------------------------------------------------------------------------
# Write Excel file
# ---------------------------------------------------------------------------
print(f"\nWriting Excel file to {OUTPUT}...")

with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
    for sheet_name, data in sheets.items():
        # Excel sheet names max 31 chars
        safe_name = sheet_name[:31]
        data.to_excel(writer, sheet_name=safe_name, index=False)
        print(f"  Written sheet: '{safe_name}' ({len(data)} rows)")

print(f"\nDone. Source_Data.xlsx saved to {OUTPUT}")
print("\nSheets written:")
for name in sheets:
    print(f"  - {name}")
