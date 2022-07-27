import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
from patsy import dmatrix
from pymc3.math import log1pexp as softplus
def softplus_np(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

class EvidenceModel(object):
    
    def __init__(self, data):
        self.data = data
        self.subject_ix, self.unique_subjects = pd.factorize(self.data.index.get_level_values('subject'))
        self.n_subjects = len(self.unique_subjects) 
        
        
    def build_model(self):

        base_numbers = self.data.n_safe.unique()
        choices = self.data.chose_risky.values

        mean_safe = np.mean(np.log(base_numbers))
        std_safe = np.std(np.log(base_numbers)) 

        self.coords = {
            "subject": self.unique_subjects,
            "presentation": ['first', 'second'],
        }

        with pm.Model(coords=self.coords) as self.model:

            inputs = self._get_model_input()
            for key, value in inputs.items():
                inputs[key] = pm.Data(key, value)

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

            # ix0 = first presented, ix1=later presented
            evidence_sd_mu = pm.HalfNormal("evidence_sd_mu", sigma=1., dims=('presentation'))
            evidence_sd_sd = pm.HalfCauchy("evidence_sd_sd", 1., dims=('presentation'))
            evidence_sd = pm.TruncatedNormal('evidence_sd',
                                              mu=evidence_sd_mu,
                                              sigma=evidence_sd_sd,
                                              lower=0,
                                              dims=('subject', 'presentation'))


            post_risky_mu, post_risky_sd = get_posterior(risky_prior_mu[inputs['subject_ix']],
                                                         risky_prior_sd[inputs['subject_ix']],
                                                         inputs['risky_mu'],
                                                         evidence_sd[inputs['subject_ix'], inputs['risky_ix']])


            post_safe_mu, post_safe_sd = get_posterior(safe_prior_mu,
                                                       safe_prior_sd,
                                                       inputs['safe_mu'],
                                                       evidence_sd[inputs['subject_ix'], inputs['safe_ix']])

            diff_mu, diff_sd = get_diff_dist(post_risky_mu, post_risky_sd, post_safe_mu, post_safe_sd)

            p = pm.Deterministic('p', cumulative_normal(tt.log(.55), diff_mu, diff_sd))


            ll = pm.Bernoulli('ll_bernoulli', p=p, observed=choices)


    def _get_model_input(self, data=None):
        
        if data is None:
            data = self.data
        
        d = {}
        d['risky_mu'] = np.log(data['n_risky']).values
        d['safe_mu'] = np.log(data['n_safe']).values

        d['risky_ix'] = (~data['risky_first']).astype(int).values
        d['safe_ix'] = (data['risky_first']).astype(int).values
        
        
        mapping = dict(zip(self.unique_subjects, range(self.n_subjects)))
        print(mapping)
        d['subject_ix'] = data.index.get_level_values('subject').map(mapping).values

        return d
            
    def sample(self, draws=1000, tune=1000, target_accept=0.95):
        
        with self.model:
            self.trace = pm.sample(draws, tune=tune, target_accept=0.95, return_inferencedata=True)
        
        return self.trace
    
    def get_predictions(self, data=None, trace=None):
        if data is None:
            data = self.make_example_data()
            
        if trace is None:
            trace = self.trace

        if not hasattr(self, 'model'):
            raise Exception('Model was not built yet! First run model.build_model()')
            
        model_input = self._get_model_input(data)
        
        with self.model:
            pm.set_data(model_input)

            posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["p"])
            model_preds = posterior_predictive["p"]
            
        model_preds = pd.DataFrame(model_preds,
                                   index=pd.Index(np.arange(len(model_preds)), name='sample'),
                                   columns=data.index)
        model_preds = pd.concat((model_preds.T, data), 1).set_index(data.columns.tolist(), append=True)
        model_preds.columns.name = 'sample'

        return model_preds.stack().to_frame('p_predicted').reset_index()
        
            
    def make_example_data(self, data=None, n=100):
        
        if data is None:
            data = self.data

        subjects = data.index.unique(level='subject')

        min_frac = data['frac'].min()
        max_frac = data['frac'].max()

        frac = np.linspace(min_frac, max_frac, n)
        risky_first = [False, True]

        safe_n = data['n_safe'].unique()

        perm = pd.MultiIndex.from_product([subjects, frac, safe_n, risky_first], names=['subject', 'frac', 'n_safe', 'risky_first']).to_frame().reset_index(drop=True)
        perm['n_risky'] = perm['n_safe'] * perm['frac']

        perm['n1'] = perm['n_risky'].where(perm['risky_first'], perm['n_safe'])
        perm['n2'] = perm['n_safe'].where(perm['risky_first'], perm['n_risky'])
        perm = perm.set_index('subject')
        
        perm['log(risky/safe)'] = np.log(perm['frac'])

        return perm   

class EvidenceModelSinglePrior(EvidenceModel):

    def build_model(self):

        choices = self.data.chose_risky.values

        self.coords = {
            "subject": self.unique_subjects,
            "presentation": ['first', 'second'],
            "risky": ['risky', 'safe']
        }

        with pm.Model(coords=self.coords) as self.model:

            inputs = self._get_model_input()
            for key, value in inputs.items():
                inputs[key] = pm.Data(key, value)

            # Hyperpriors for group nodes
            prior_mu_mu = pm.HalfNormal("prior_mu_mu", sigma=np.log(20.))
            prior_mu_sd = pm.HalfCauchy('prior_mu_sd', .5)
            prior_mu_offset = pm.Normal('prior_mu_offset', mu=0, sd=1, dims='subject')#shape=n_subjects)
            prior_mu = pm.Deterministic('prior_mu', prior_mu_mu + prior_mu_sd * prior_mu_offset,
                                             dims='subject')

            prior_sd_mu = pm.HalfNormal("prior_sd_mu", sigma=1.25)
            prior_sd_sd = pm.HalfCauchy('prior_sd_sd', .5)

            prior_sd = pm.TruncatedNormal('prior_sd',
                                           mu=prior_sd_mu,
                                          sigma=prior_sd_sd,
                                          lower=0,
                                          dims='subject')

            # ix0 = first presented, ix1=later presented
            evidence_sd_mu = pm.HalfNormal("evidence_sd_mu", sigma=1., dims=('presentation', 'risky'))
            evidence_sd_sd = pm.HalfCauchy("evidence_sd_sd", 1., dims=('presentation', 'risky'))
            evidence_sd = pm.TruncatedNormal('evidence_sd',
                                              mu=evidence_sd_mu,
                                              sigma=evidence_sd_sd,
                                              lower=0,
                                              dims=('subject', 'presentation', 'risky'))


            post_risky_mu, post_risky_sd = get_posterior(prior_mu[inputs['subject_ix']],
                                                         prior_sd[inputs['subject_ix']],
                                                         inputs['risky_mu'],
                                                         evidence_sd[inputs['subject_ix'], inputs['risky_ix'], 0])


            post_safe_mu, post_safe_sd = get_posterior(prior_mu[inputs['subject_ix']],
                                                       prior_sd[inputs['subject_ix']],
                                                       inputs['safe_mu'],
                                                       evidence_sd[inputs['subject_ix'], inputs['safe_ix'], 1])

            diff_mu, diff_sd = get_diff_dist(post_risky_mu, post_risky_sd, post_safe_mu, post_safe_sd)

            p = pm.Deterministic('p', cumulative_normal(tt.log(.55), diff_mu, diff_sd))

            ll = pm.Bernoulli('ll_bernoulli', p=p, observed=choices)


class EvidenceModelRegression(EvidenceModel):
    
    def __init__(self, data, regressors=None):
        self.data = data
        self.subject_ix, self.unique_subjects = pd.factorize(self.data.index.get_level_values('subject'))
        self.n_subjects = len(self.unique_subjects) 
        
        self.design_matrices = {}
        
        if regressors is None:
            regressors = {}
        
        for par in ['evidence_sd1', 'evidence_sd2', 'risky_prior_mu', 'risky_prior_sd']:
            
            if par not in regressors:
                regressors[par] = '1'
            
            self.design_matrices[par] = dmatrix(regressors[par], self.data)
    
    def build_model(self):

        base_numbers = self.data.n_safe.unique()
        choices = self.data.chose_risky.values

        safe_prior_mu = np.mean(np.log(base_numbers))
        safe_prior_sd = np.std(np.log(base_numbers)) 

        self.coords = {
            "subject": self.unique_subjects,
            "presentation": ['first', 'second'],
            "risky_prior_mu_regressors":self.design_matrices['risky_prior_mu'].design_info.term_names,
            "risky_prior_sd_regressors":self.design_matrices['risky_prior_sd'].design_info.term_names,
            "evidence_sd1_regressors":self.design_matrices['evidence_sd1'].design_info.term_names,
            "evidence_sd2_regressors":self.design_matrices['evidence_sd2'].design_info.term_names            
            
        }
        
                              
                                              
        with pm.Model(coords=self.coords) as self.model:

            inputs = self._get_model_input()
            for key, value in inputs.items():
                inputs[key] = pm.Data(key, value)
                
            def build_hierarchical_nodes(name, mu_intercept=0.0, sigma=.5):
                nodes = {}

                mu = np.zeros(self.design_matrices[name].shape[1])
                mu[0] = mu_intercept

                nodes[f'{name}_mu'] = pm.Normal(f"{name}_mu", 
                                              mu=mu, 
                                              sigma=sigma,
                                              dims=f'{name}_regressors')
                nodes[f'{name}_sd'] = pm.HalfCauchy(f'{name}_sd', .5, dims=f'{name}_regressors')
                nodes[f'{name}_offset'] = pm.Normal(f'{name}_offset', mu=0, sd=1, dims=('subject', f'{name}_regressors'))
                nodes[name] = pm.Deterministic(name, nodes[f'{name}_mu'] + nodes[f'{name}_sd'] * nodes[f'{name}_offset'],
                                              dims=('subject', f'{name}_regressors'))
                
                nodes[f'{name}_trialwise'] = softplus(tt.sum(nodes[name][inputs['subject_ix']] * \
                                                               np.asarray(self.design_matrices[name]), 1))
                
                return nodes

            # Hyperpriors for group nodes
            
            nodes = {}
            
            nodes.update(build_hierarchical_nodes('risky_prior_mu'), mu_intercept=np.log(20.))
            nodes.update(build_hierarchical_nodes('risky_prior_sd'), mu_intercept=1.)
            nodes.update(build_hierarchical_nodes('evidence_sd1'), mu_intercept=1.)
            nodes.update(build_hierarchical_nodes('evidence_sd2'), mu_intercept=1.)
            
            evidence_sd = tt.stack((nodes['evidence_sd1_trialwise'], nodes['evidence_sd2_trialwise']), 0)
        
            

            post_risky_mu, post_risky_sd = get_posterior(nodes['risky_prior_mu_trialwise'],
                                                         nodes['risky_prior_sd_trialwise'],
                                                         inputs['risky_mu'],
                                                         evidence_sd[inputs['risky_ix'], np.arange(self.data.shape[0])])


            post_safe_mu, post_safe_sd = get_posterior(safe_prior_mu,
                                                       safe_prior_sd,
                                                       inputs['safe_mu'],
                                                       evidence_sd[inputs['safe_ix'], np.arange(self.data.shape[0])])

            diff_mu, diff_sd = get_diff_dist(post_risky_mu, post_risky_sd, post_safe_mu, post_safe_sd)

            p = pm.Deterministic('p', cumulative_normal(tt.log(.55), diff_mu, diff_sd))


            ll = pm.Bernoulli('ll_bernoulli', p=p, observed=choices)

def get_posterior(mu1, sd1, mu2, sd2):

    var1, var2 = sd1**2, sd2**2

    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))

def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, tt.sqrt(sd1**2+sd2**2)

def cumulative_normal(x, mu, sd, s=np.sqrt(2.)):
#     Cumulative distribution function for the standard normal distribution
    return tt.clip(0.5 + 0.5 *
                   tt.erf((x - mu) / (sd*s)), 1e-9, 1-1e-9)
