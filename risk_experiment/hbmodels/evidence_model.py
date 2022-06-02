import arviz as az
import os
import os.path as op
import pymc3 as pm
import numpy as np
import pandas as pd
from risk_experiment.utils import get_all_task_behavior
import theano.tensor as tt

def get_posterior(mu1, sd1, mu2, sd2):

    var1, var2 = sd1**2, sd2**2

    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))

def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, tt.sqrt(sd1**2+sd2**2)

def cumulative_normal(x, mu, sd, s=np.sqrt(2.)):
#     Cumulative distribution function for the standard normal distribution
    return tt.clip(0.5 + 0.5 *
                   tt.erf((x - mu) / (sd*s)), 1e-9, 1-1e-9)

bids_folder = '/data'

target_folder = op.join(bids_folder, 'derivatives', 'evidence_models',)

if not op.exists(target_folder):
    os.makedirs(target_folder)

df = get_all_task_behavior(bids_folder=bids_folder)

df.loc[df.risky_first, 'base_number'] = df['n2']
df.loc[~df.risky_first, 'base_number'] = df['n1']

risky_mu = np.log(np.where(df['prob1'] == 1.0, df['n2'], df['n1']))
safe_mu = np.log(np.where(df['prob1'] == 1.0, df['n1'], df['n2']))
choices = df['chose_risky']
risky_first = df['risky_first'].values

# ix0 = first presented, ix1=later presented
risky_ix = (~risky_first).astype(int)
safe_ix = (risky_first).astype(int)

unique_subjects = df.index.unique(level='subject')
n_subjects = len(unique_subjects)
subject_ix = df.index.codes[0]
coords = {
    "subject": unique_subjects,
    "presentation": ['first', 'second'],
    "obs_id":df.index,
}
base_numbers = [5., 7., 10., 14., 20., 28.]
mean_safe = np.mean(np.log(base_numbers))
std_safe = np.std(np.log(base_numbers)) 

with pm.Model(coords=coords) as model:

    # Hyperpriors for group nodes
    risky_prior_mu_mu = pm.HalfNormal("risky_prior_mu_mu", sigma=np.log(20.))
    risky_prior_mu_sd = pm.HalfCauchy('risky_prior_mu_sd', .5)
    risky_prior_mu_offset = pm.Normal('risky_prior_mu_offset', mu=0, sd=1, dims='subject')#shape=n_subjects)
    risky_prior_mu = pm.Deterministic('risky_prior_mu', risky_prior_mu_mu + risky_prior_mu_sd * risky_prior_mu_offset,
                                     dims='subject')

    risky_prior_sd_mu = pm.HalfNormal("risky_prior_sd_mu", sigma=1.25)
    risky_prior_sd_sd = pm.HalfCauchy('risky_prior_sd_sd', .5)

    risky_prior_sd = pm.TruncatedNormal('risky_prior_sd',
                                        mu=risky_prior_sd_mu,
                                        sigma=risky_prior_sd_sd,
                                        lower=0,
                                        dims='subject')

    safe_prior_mu = mean_safe
    safe_prior_sd = std_safe


    #     # ix0 = first presented, ix1=later presented
    evidence_sd_mu = pm.HalfNormal("evidence_sd_mu", sigma=1., dims=('presentation'))
    evidence_sd_sd = pm.HalfCauchy("evidence_sd_sd", 1., dims=('presentation'))
    evidence_sd = pm.TruncatedNormal('evidence_sd',
                                      mu=evidence_sd_mu,
                                      sigma=evidence_sd_sd,
                                      lower=0,
                                      dims=('subject', 'presentation'))


    post_risky_mu, post_risky_sd = get_posterior(risky_prior_mu[subject_ix],
                                                 risky_prior_sd[subject_ix],
                                                 risky_mu,
                                                 evidence_sd[subject_ix, risky_ix])


    post_safe_mu, post_safe_sd = get_posterior(safe_prior_mu,
                                               safe_prior_sd,
                                               safe_mu,
                                               evidence_sd[subject_ix, safe_ix])

    diff_mu, diff_sd = get_diff_dist(post_risky_mu, post_risky_sd, post_safe_mu, post_safe_sd)

    p = cumulative_normal(tt.log(.55), diff_mu, diff_sd)


    ll = pm.Bernoulli('ll_bernoulli', p=p, observed=choices)



with model:
    trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)

az.to_netcdf(trace,  op.join(target_folder, 'evidence_model-1.nc'))
