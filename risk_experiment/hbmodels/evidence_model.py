import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt

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
            self.trace = pm.sample(draws, tune=500, target_accept=0.95, return_inferencedata=True)
        
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

def get_posterior(mu1, sd1, mu2, sd2):

    var1, var2 = sd1**2, sd2**2

    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))

def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, tt.sqrt(sd1**2+sd2**2)

def cumulative_normal(x, mu, sd, s=np.sqrt(2.)):
#     Cumulative distribution function for the standard normal distribution
    return tt.clip(0.5 + 0.5 *
                   tt.erf((x - mu) / (sd*s)), 1e-9, 1-1e-9)
