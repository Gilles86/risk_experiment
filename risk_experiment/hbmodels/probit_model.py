import warnings
import arviz as az
import pandas as pd
import numpy as np
import bambi as bmb

bids_folder = '/data'


class ProbitModel(object):

    models = {'model1': 'chose_risky ~ 1 + C(n_safe)*x*risky_first + (C(n_safe)*x*risky_first|subject)'}

    def __init__(self, data, model_type, bids_folder='/data'):
        self.bids_folder = bids_folder
        self.data = data

        self.data['x'] = self.data['log(risky/safe)']

        if self.data.choice.isnull().any():
            warnings.warn('Removing null choices')
            self.data = self.data[self.data.choice.isnull()]
        self.model_type = model_type

    def build_model(self, bids_folder='/data/ds-tmsrisk'):

        formula = self.models[self.model_type]

        self.model = bmb.Model(formula, self.data[[
                               'chose_risky', 'x', 'risky_first', 'n_safe']].reset_index(), family="bernoulli", link='probit')

    def create_test_matrix(self, empirical=True):

        if empirical:
            df = self.data.copy()
            df['x'] = df.groupby(['subject'])['x'].apply(
                lambda d: pd.qcut(d, 6, duplicates='drop')).apply(lambda x: x.mid, 1)

            return df.groupby(['subject', 'risky_first', 'session', 'x', 'n_safe']).size().to_frame('size').reset_index().drop('size', 1)

        else:
            unique_subjects = self.data.index.unique(level='subject')

            risky_safe = np.linspace(np.log(1), np.log(4), 20)
            risky_first = [True, False]
            n_safe = self.data.n_safe.unique().astype(np.float64)
            session = ['3t2', '7t2']

            d = pd.DataFrame(np.array([e.ravel() for e in np.meshgrid(unique_subjects, risky_safe, risky_first, n_safe, session)]).T,
                             columns=['subject', 'x', 'risky_first', 'n_safe', 'session'])

            d['risky_first'] = d['risky_first'].astype(bool)

            return d

    def get_group_nlc_parameters(self, trace):
        risky_first = [True, False]
        x = [0, 1]
        n_safe = self.data['n_safe'].unique()

    def sample(self, draws=1000, tune=1000, target_accept=.85):
        trace = self.model.fit(
            draws, tune, target_accept=target_accept, init='adapt_diag')
        return trace

    def get_predictions(self, trace, empirical=True, return_summary_stats=True, thin=5):
        test_data = self.create_test_matrix(empirical=empirical)
        test_data.index.name = 'test_values'
        pred = self.model.predict(trace, 'mean', test_data, inplace=False,)[
            'posterior']['chose_risky_mean'].to_dataframe()
        pred.index = pred.index.set_names('test_values', -1)
        pred = pred.join(test_data).loc[(
            slice(None), slice(None, None, thin)), :]

        if return_summary_stats:
            m = pred.groupby(['subject', 'x', 'risky_first', 'n_safe'])[
                ['chose_risky_mean']].mean()
            ci = pred.groupby(['subject', 'x', 'risky_first', 'n_safe'])['chose_risky_mean'].apply(lambda x: pd.Series(az.hdi(x.values),
                                                                                                                       index=['lower', 'higher'])).unstack()

            m = m.join(ci)
            return m

        else:
            return pred
