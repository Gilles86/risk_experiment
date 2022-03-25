import argparse
import bambi as bmb
import os.path as op
import pandas as pd
from risk_experiment.utils import get_task_paradigm
import arviz as az
from scipy.stats import zscore


# def main(parameter, session, mask, bids_folder):
# bids_folder = '/data'
# session = '3t2'
# parameter = 'sd'
# mask = 'wang15_ips'

bad_subjects = ['03', '32']

models = {'model1':'{parameter} ~ 1 + log_n1*chose_risky + chose_risky*risky_first + log_n1*risky_first + (log_n1*chose_risky|subject) + (chose_risky*risky_first|subject) + (log_n1*risky_first|subject)',
        'model2': '{parameter} ~ 1 + chose_risky*risky_first + (risky_first*chose_risky|subject)',
        'model3': '{parameter} ~ 1 + chose_first*chose_risky + (chose_first*chose_risky|subject)'}


def main(parameter, session, mask, bids_folder, model='model1'):
    E = pd.read_csv(op.join(bids_folder, 'derivatives', 'decoded_pdfs.volume', f'group_mask-{mask}_pars.tsv'), sep='\t', dtype={'subject':str}).set_index(['subject', 'session', 'trial_nr'])

    E = E.xs(session, 0, 'session')
    print(E)

    df = get_task_paradigm(bids_folder=bids_folder, session=session)
    df = df.xs(session, 0, 'session').droplevel('run', 0)

    df = df.join(E.drop(['n1', 'prob1'], 1))

    df.drop(bad_subjects, level='subject')
    print(df.shape)
    df = df[~df.choice.isnull()]

    df['x'] = df['log(risky/safe)']
    df['log_n1'] = df['log(n1)']
    df['log_n1'] = df.groupby(['subject']).log_n1.apply(zscore)
    df['chose_first'] = (df['choice'] -2)*-1

    # formula = f'{parameter} ~ 1 + x + {parameter} + {parameter}:x + (x*session|subject)+ ({parameter}|subject) + ({parameter}:x|subject)'
    formula = models[model].format(**locals())

    glm = bmb.Model(formula, df[[parameter, 'log_n1', 'risky_first', 'chose_risky', 'chose_first']].reset_index())
    print(glm)

    results = glm.fit(2000, 2000, target_accept=.85)

    az.to_netcdf(results, op.join(bids_folder, 'derivatives', 'probit_models', f'group_model-{model}_ses-{session}_mask-{mask}_parameter-{parameter}.nc'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('mask', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    args = parser.parse_args()

    main(args.parameter, args.session, args.mask, args.bids_folder, model=args.model)
