import re
import numpy as np
import argparse
from risk_experiment.utils.data import get_all_behavior, get_all_subjects
import os.path as op
import os
import arviz as az
import bambi
import pandas as pd

def main(model_label, session, burnin=1000, samples=1000, bids_folder='/data/ds-risk'):

    print(model_label)
    df = get_data(model_label, session, bids_folder)
    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.9

    model = build_model(model_label, df, session, bids_folder)
    trace = model.fit(burnin, samples, init='adapt_diag', target_accept=target_accept)
    az.to_netcdf(trace,
                 op.join(target_folder, f'model-{model_label}_ses-{session}_trace.netcdf'))

def build_model(model_label, df, session=None, bids_folder='/data/ds-risk'):
    if model_label == 'probit_simple':
        model = bambi.Model('chose_risky ~ x  + (x|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label == 'probit_order':
        model = bambi.Model('chose_risky ~ x*risky_first  + (x*risky_first|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label == 'probit_full':
        model = bambi.Model('chose_risky ~ x*risky_first*n_safe  + (x*risky_first*n_safe|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural1'):
        model = bambi.Model('chose_risky ~ x*risky_first*n_safe + x*sd  + (x*risky_first*n_safe + sd*x|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural2'):
        model = bambi.Model('chose_risky ~ x*risky_first*n_safe + x*sd  + (x*risky_first*n_safe + sd*x|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural3'):
        model = bambi.Model('chose_risky ~ x*risky_first*n_safe + x*median_split_sd  + (x*risky_first*n_safe + x*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural4'):
        model = bambi.Model('chose_risky ~ x*sd  + (x*sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural5'):
        model = bambi.Model('chose_risky ~ x*risky_first*n_safe*median_split_sd  + (x*risky_first*n_safe*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural6'):
        model = bambi.Model('chose_risky ~ x*risky_first*n_safe*median_split_sd  + (x*risky_first*n_safe*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural7'):
        model = bambi.Model('chose_risky ~ x*median_split_sd  + (x*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural8'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_sd  + (x*risky_first*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural9'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_sd  + (x*risky_first|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural10'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_sd*risk_preference  + (x*risky_first*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural107'):
        model = bambi.Model('chose_risky ~ x*median_split_sd*risk_preference  + (x*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_neural1072'):
        model = bambi.Model('chose_risky ~ x*sd*risk_preference  + (x*sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label.startswith('probit_pupil_median_split'):
        model = bambi.Model('chose_risky ~ x*median_split_pupil  + (x*median_split_pupil|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_pupil'):
        model = bambi.Model('chose_risky ~ x*pupil  + (x*pupil|subject)', df.reset_index(), link='probit', family='bernoulli')
    # if model_label.startswith('probit_pupil1'):
    #     model = bambi.Model('chose_risky ~ x*median_split_pupil_baseline  + (x*median_split_pupil_baseline|subject)', df.reset_index(), link='probit', family='bernoulli')
    # if model_label.startswith('probit_pupil2'):
    #     model = bambi.Model('chose_risky ~ x*risky_first*median_split_pupil_baseline  + (x*risky_first*median_split_pupil_baseline|subject)', df.reset_index(), link='probit', family='bernoulli')
    # if model_label.startswith('probit_pupil3'):
    #     model = bambi.Model('chose_risky ~ x*risky_first*pupil  + (x*risky_first*pupil|subject)', df.reset_index(), link='probit', family='bernoulli')
    # if model_label.startswith('probit_pupil4'):
    #     model = bambi.Model('chose_risky ~ x*risky_first*"q(pupil)"  + (x*risky_first*"q(pupil)"|subject)', df.reset_index(), link='probit', family='bernoulli')

    return model

def get_data(model_label, session, bids_folder='/data/ds-risk', drop_outliers=False):

    if model_label.endswith('_no_outliers'):
        print('no outliers!')
        drop_outliers = True

    df = get_all_behavior(sessions=session, bids_folder=bids_folder, drop_outliers=drop_outliers)
    df['x'] = df['log(risky/safe)']

    if (model_label == 'probit_neural1') or model_label.startswith('probit_neural3') or model_label.startswith('probit_neural4') or model_label.startswith('probit_neural5'):
        decoding_info = pd.concat([sub.get_decoding_info(session, mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
        df = df.join(decoding_info)
        df['median_split(sd)'] = df.groupby(['subject', 'session'], group_keys=False)['sd'].apply(lambda d: d>d.quantile())
        df['median_split_sd'] = df['median_split(sd)']
    elif model_label.startswith('probit_neural2'):
        decoding_info = pd.concat([sub.get_decoding_info(session, mask='npcr', n_voxels=100) for sub in get_all_subjects(bids_folder)])
        df = df.join(decoding_info)
        df['median_split(sd)'] = df.groupby(['subject', 'session'], group_keys=False)['sd'].apply(lambda d: d>d.quantile())
    elif model_label.startswith('probit_neural6') or model_label.startswith('probit_neural7') or model_label.startswith('probit_neural8') or model_label.startswith('probit_neural9'):
        decoding_info = pd.concat([sub.get_decoding_info(session, mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
        df = df.join(decoding_info)
        df['median_split(sd)'] = df.groupby(['subject', 'session', 'n1', 'n2'], group_keys=False)['sd'].apply(lambda d: d>d.quantile())
        df['median_split_sd'] = df['median_split(sd)']
    elif model_label.startswith('probit_neural10'):
        decoding_info = pd.concat([sub.get_decoding_info(session, mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
        df = df.join(decoding_info)
        df['median_split(sd)'] = df.groupby(['subject', 'session', 'n1', 'n2'], group_keys=False)['sd'].apply(lambda d: d>d.quantile())
        df['median_split_sd'] = df['median_split(sd)']
        rnp = pd.read_csv(op.join(bids_folder, 'derivatives', 'cogmodels', 'gamma_rnp_simple.tsv'), sep='\t', index_col=[0,1], dtype={'subject':str})[['rnp']]
        rnp['risk_preference'] = (rnp['rnp'] > 0.55).map({True:'risk seeking', False:'risk averse'})
        df = df.join(rnp)

    elif model_label.startswith('probit_pupil'):
        reg = re.compile('probit_pupil.*_(?P<variable>n1_pre|n1_post|n1_prepost|n2_choice)')
        variable = reg.match(model_label).group(1)
        pupil = pd.read_csv(op.join(bids_folder, 'derivatives', 'pupil', 'model-n1_n2_n', 'pupil_pre_post12.tsv'), sep='\t', index_col=[0,1,2,3])
        pupil['q(pupil)'] = pupil.groupby(['subject', 'event type', 'prepost'], group_keys=False)['pupil'].apply(lambda x: pd.qcut(x, 5, labels=['q1', 'q2', 'q3', 'q4', 'q5']))
        if variable == 'n1_pre':
            pupil = pupil.xs('n1', 0, 'event type').xs('pre', 0, 'prepost')
        elif variable == 'n1_post':
            pupil = pupil.xs('n1', 0, 'event type').xs('post', 0, 'prepost')
        elif variable == 'n1_prepost':
            pupil = pupil.xs('n1', 0, 'event type').xs('prepost', 0, 'prepost')
        elif variable == 'n2_choice':
            pupil = pupil.xs('n2', 0, 'event type')
        df = df.join(pupil)
        df['median_split_pupil'] = df.groupby(['subject'], group_keys=False)['pupil'].apply(lambda d: d>d.quantile()).map({True:'High pupil dilation', False:'Low pupil dilation'})
        print(df)

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label')
    parser.add_argument('session')
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    args = parser.parse_args()

    main(args.model_label, args.session, bids_folder=args.bids_folder)



