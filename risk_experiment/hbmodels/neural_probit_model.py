import argparse
import bambi as bmb
import os.path as op
import pandas as pd
from risk_experiment.utils import get_all_task_behavior
import arviz as az
from scipy.stats import zscore


# def main(parameter, session, mask, bids_folder):
# bids_folder = '/data'
# session = '3t2'
# parameter = 'sd'
# mask = 'wang15_ips'

bad_subjects = ['03', '32']

models = {'model1': 'chose_risky ~ 1 + x*sd + x*risky_first + sd*risky_first + (x*sd|subject) + (x*risky_first|subject) + (sd*risky_first|subject)',
          'model2': 'chose_risky ~ 1 + log_n1*sd + log_n1*risky_first + log_n1*risky_first + (log_n1*sd|subject) + (log_n1*risky_first|subject) + (sd*risky_first|subject)',
          'model3': 'chose_risky ~ 1 + x*sd*risky_first + (x*sd*risky_first|subject)'}



def main(session, mask, bids_folder, model='model1'):
    E = pd.read_csv(op.join(bids_folder, 'derivatives', 'decoded_pdfs.volume', f'group_mask-{mask}_pars.tsv'), sep='\t', dtype={'subject':str}).set_index(['subject', 'session', 'trial_nr'])

    E = E.xs(session, 0, 'session')

    df = get_all_task_behavior(bids_folder=bids_folder, session=session)
    df = df.xs(session, 0, 'session').droplevel('run', 0)
    df = df.join(E.drop(['n1', 'prob1', 'log(n1)'], 1))
    print(df)

    df = df.drop(bad_subjects, level='subject')
    print(df.shape)
    df = df[~df.choice.isnull()]
    print(df)

    df['x'] = df['log(risky/safe)']
    df['log_n1'] = df['log(n1)']
    df['chose_first'] = (df['choice'] -2)*-1

    df['sd'] = df.groupby(['subject'])['sd'].apply(zscore)

    formula = models[model].format(**locals())

    glm = bmb.Model(formula, df[['sd', 'risky_first', 'chose_risky', 'x', 'log_n1']].reset_index(), family='bernoulli', link='probit')
    print(glm)

    results = glm.fit(2000, 2000, target_accept=.85, init='adapt_diag')

    az.to_netcdf(results, op.join(bids_folder, 'derivatives', 'probit_models', f'group_model-{model}_ses-{session}_mask-{mask}_parameter-neuralprobit.nc'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('session', default=None)
    parser.add_argument('mask', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.session, args.mask, args.bids_folder, model=args.model)