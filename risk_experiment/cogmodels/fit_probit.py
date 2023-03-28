import re
import numpy as np
import argparse
from risk_experiment.utils.data import get_all_behavior, get_all_subjects
import os.path as op
import os
import arviz as az
import bambi
import pandas as pd

def main(model_label, session, burnin=1000, samples=1000, bids_folder='/data/ds-risk', n_cores=4, roi=None):

    print(model_label)
    df = get_data(model_label, session, bids_folder, roi=roi)
    print(df)
    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.9

    model = build_model(model_label, df, session, bids_folder, roi)
    trace = model.fit(burnin, samples, init='adapt_diag', target_accept=target_accept, cores=n_cores)
    if session is None:
        az.to_netcdf(trace,
                    op.join(target_folder, f'model-{model_label}_trace.netcdf'))
    else:
        if roi is None:
            az.to_netcdf(trace,
                        op.join(target_folder, f'model-{model_label}_ses-{session}_trace.netcdf'))
        else:
            az.to_netcdf(trace,
                        op.join(target_folder, f'model-{model_label}_ses-{session}_roi-{roi}_trace.netcdf'))


def build_model(model_label, df, session=None, bids_folder='/data/ds-risk', roi=None):
    if model_label == 'probit_simple':
        model = bambi.Model('chose_risky ~ x  + (x|subject)', df.reset_index(), link='probit', family='bernoulli')
    if model_label == 'probit_session':
        model = bambi.Model('chose_risky ~ x*session  + (x*session|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_order':
        model = bambi.Model('chose_risky ~ x*risky_first  + (x*risky_first|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_full':
        model = bambi.Model('chose_risky ~ x*risky_first*n_safe  + (x*risky_first*n_safe|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_neural1':
        model = bambi.Model('chose_risky ~ x*risky_first*C(n_safe)*sd + (x*risky_first*C(n_safe)*sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_neural2'):
        model = bambi.Model('chose_risky ~ x*risky_first*C(n_safe)*sd + (x*risky_first*C(n_safe)*sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_neural3'):
        model = bambi.Model('chose_risky ~ x*risky_first*C(n_safe)*median_split_sd + (x*risky_first*C(n_safe)*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_neural4'):
        model = bambi.Model('chose_risky ~ x*median_split_sd + (x*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_neural5'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_sd + (x*risky_first*median_split_sd|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_neural6'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_sd*session + (x*risky_first*median_split_sd*session|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_neural7'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_sd*C(n_safe)*session + (x*risky_first*median_split_sd*C(n_safe)*session|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_neural8'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_sd + x*risky_first*session + (x*risky_first*median_split_sd + x*risky_first*session|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_neural9'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_sd*C(n_safe) + x*risky_first*C(n_safe)*session + (x*risky_first*median_split_sd*C(n_safe) + x*risky_first*C(n_safe)*session|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_neural10'):
        model = bambi.Model('chose_risky ~ x*median_split_sd+session*x + (x*median_split_sd+session*x|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_pupil_median_split1'):
        model = bambi.Model('chose_risky ~ x*median_split_pupil  + (x*median_split_pupil|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_pupil_median_split2'):
        model = bambi.Model('chose_risky ~ x*C(n_safe)*risky_first*median_split_pupil  + (x*median_split_pupil|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_pupil_median_split3'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_pupil  + (x*risky_first*median_split_pupil|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_pupil'):
        model = bambi.Model('chose_risky ~ x*pupil  + (x*pupil|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_subcortical_response1'):
        model = bambi.Model('chose_risky ~ x*median_split_subcortical_response  + (x*median_split_subcortical_response|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_subcortical_response2'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_subcortical_response  + (x*risky_first*median_split_subcortical_response|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label.startswith('probit_subcortical_response3'):
        model = bambi.Model('chose_risky ~ x*risky_first*median_split_subcortical_response*C(n_safe)  + (x*risky_first*median_split_subcortical_response*C(n_safe)|subject)', df.reset_index(), link='probit', family='bernoulli')

    return model

def get_data(model_label, session, bids_folder='/data/ds-risk', drop_outliers=False, roi=None):

    if model_label.endswith('_no_outliers'):
        print('no outliers!')
        drop_outliers = True

    df = get_all_behavior(sessions=session, bids_folder=bids_folder, drop_outliers=drop_outliers)
    df['x'] = df['log(risky/safe)']

    if model_label.startswith('probit_neural'):
        if session is None:
            decoding_info_3t = pd.concat([sub.get_decoding_info('3t2', mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
            decoding_info_7t = pd.concat([sub.get_decoding_info('7t2', mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
            decoding_info = pd.concat((decoding_info_3t, decoding_info_7t))
        else:
            decoding_info = pd.concat([sub.get_decoding_info(session, mask='npcr', n_voxels=0.0) for sub in get_all_subjects(bids_folder)])
        df = df.join(decoding_info)
        df['median_split(sd)'] = df.groupby(['subject', 'session', 'n1'], group_keys=False)['sd'].apply(lambda d: d>d.quantile())
        df['median_split_sd'] = df['median_split(sd)']

    elif model_label.startswith('probit_pupil'):
        reg = re.compile('probit_pupil.*_(?P<variable>n1_pre|n1_post|n1_prepost|n2_choice)$')
        variable = reg.match(model_label).group(1)
        pupil = pd.read_csv(op.join(bids_folder, 'derivatives', 'pupil', 'model-n1_n2_n', 'pupil_pre_post12.tsv'), sep='\t', index_col=[0,1,2,3], dtype={'subject':str})
        pupil['q(pupil)'] = pupil.groupby(['subject', 'event type', 'prepost'], group_keys=False)['pupil'].apply(lambda x: pd.qcut(x, 5, labels=['q1', 'q2', 'q3', 'q4', 'q5']))
        if variable == 'n1_pre':
            print('N1_PRE')
            pupil = pupil.xs('n1', 0, 'event type').xs('pre', 0, 'prepost')
        elif variable == 'n1_post':
            print('N1_POST')
            print('YO')
            pupil = pupil.xs('n1', 0, 'event type').xs('post', 0, 'prepost')
        elif variable == 'n1_prepost':
            print('N1_PREPOST')
            pupil = pupil.xs('n1', 0, 'event type').xs('post-pre', 0, 'prepost')
        elif variable == 'n2_choice':
            print('N2_CHOICE')
            pupil = pupil.xs('n2', 0, 'event type')
        df = df.join(pupil)
        df['median_split_pupil'] = df.groupby(['subject'], group_keys=False)['pupil'].apply(lambda d: d>d.quantile()).map({True:'High pupil dilation', False:'Low pupil dilation'})
        # print(df)

    if model_label.startswith('probit_subcortical_response'):
        if roi is None:
            raise Exception('Need to define ROI!')
        subjects = get_all_subjects(bids_folder)
        roi_responses = pd.concat([sub.get_roi_timeseries(session, roi, single_trial=True) for sub in subjects])
        df = df.join(roi_responses)
        df['median_split_subcortical_response'] = df.groupby(['subject', 'n1'], group_keys=False)[roi].apply(lambda d: d>d.quantile()).map({True:'High activation', False:'Low activation'})
        # print(df)

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label')
    parser.add_argument('session', nargs='?', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-risk')
    parser.add_argument('--n_cores', default=4, type=int)
    parser.add_argument('--roi', default=None)
    args = parser.parse_args()

    main(args.model_label, args.session, bids_folder=args.bids_folder, n_cores=args.n_cores, roi=args.roi)



