import re
import pandas as pd
import hedfpy
import os.path as op
import os
from risk_experiment.utils import run_main
from glob import glob

analysis_params = {
                'sample_rate' : 500.0,
                'lp' : 6.0,
                'hp' : 0.01,
                'normalization' : 'zscore',
                'regress_blinks' : True,
                'regress_sacs' : True,
                'regress_xy' : False,
                'use_standard_blinksac_kernels' : True,
                }

def main(subject, session, bids_folder):


    source_folder = op.join(bids_folder, 'sourcedata', f'sub-{subject}', 'behavior', f'ses-{session}', )
    target_folder = op.join(bids_folder, 'derivatives', 'pupil_preproc', f'sub-{subject}', f'ses-{session}', 'func')

    if not op.exists(target_folder):
        os.makedirs(target_folder)


    hdf5_file = op.join(target_folder, f'sub-{subject}_pupil.hdf5')
    if op.exists(hdf5_file):
        os.remove(hdf5_file)

    ho = hedfpy.HDFEyeOperator(hdf5_file)

    # if subject in [32]:
    #     analysis_params['sample_rate'] = 1000

    for run in range(1, 9):

        hedf_key = f'sub-{subject}_run-{run}'
        fn = op.join(source_folder, f'sub-{subject}_ses-{session}_task-task_run-{run}.edf')
        ho.add_edf_file(fn)
        ho.edf_message_data_to_hdf(hedf_key)

        if (subject == '32') & (run == 1):
            analysis_params['sample_rate'] = 1000
        else:
            analysis_params['sample_rate'] = 500

        ho.edf_gaze_data_to_hdf(alias=hedf_key,
                                sample_rate=analysis_params['sample_rate'],
                                pupil_lp=analysis_params['lp'],
                                pupil_hp=analysis_params['hp'],
                                normalization=analysis_params['normalization'],
                                regress_blinks=analysis_params['regress_blinks'],
                                regress_sacs=analysis_params['regress_sacs'],
                                use_standard_blinksac_kernels=analysis_params['use_standard_blinksac_kernels'],
                                )


        properties = ho.block_properties(hedf_key)
        assert(analysis_params['sample_rate'] == properties.loc[0, 'sample_rate']), print(analysis_params['sample_rate'], properties.loc[0, 'sample_rate'])

        # # Detect behavioral messages
        messages = pd.DataFrame(ho.edf_operator.read_generic_events())

        reg = re.compile('start_type-(?P<type>stim|response|pulse)_trial-(?P<trial>[0-9.]+)_phase-(?P<phase>[0-9]+)(_key-(?P<key>.))?(_time-(?P<time>[0-9.]+))?')

        for key in ['type', 'trial', 'phase', 'key', 'time']:
            messages[key] = messages.message.apply(lambda x: reg.match(x).group(key) if reg.match(x) else None)
        start_ix = messages[messages.key == 't'].index[0]
        start_ts = messages[messages.key == 't'].iloc[0]['EL_timestamp']
        last_ts = messages.EL_timestamp.max()

        print(messages)

        events = messages.loc[start_ix+2:]

        events['onset'] = (events['EL_timestamp'] - start_ts) / 1000

        # # detect saccades
        saccades = ho.detect_saccades_during_period([start_ts, last_ts+5000], hedf_key)
        saccades['onset'] = (saccades['start_timestamp'] - start_ts) / 1000.
        print(saccades)
        # saccades['duration'] /=  1000.
        # saccades = saccades[['duration', 'onset']]

        saccades_eyelink = ho.saccades_from_message_file_during_period([start_ts, last_ts+5000], hedf_key)
        saccades_eyelink['onset'] = (saccades_eyelink['start_timestamp'] - start_ts) / 1000.

        # # Detect blinks
        blinks = ho.blinks_during_period([start_ts, last_ts + 5000], hedf_key)
        blinks['onset'] = (blinks['start_timestamp'] - start_ts) / 1000.
        blinks['duration'] /=  1000.
        blinks = blinks[['onset', 'duration']]

        eye = ho.block_properties(hedf_key).loc[0, 'eye_recorded']


        # # Get data
        d = ho.data_from_time_period([start_ts, last_ts+5000], hedf_key)
        d['time'] = (d['time'] - start_ts) / 1000. - 1./analysis_params['sample_rate']
        d = d.set_index(pd.Index(d['time'], name='time'))
        d['interpolated'] = d[f'{eye}_interpolated_timepoints'].astype(bool)
        d['pupil'] = d[f'{eye}_pupil_bp']
        d = d[['interpolated', 'pupil']]

        # # Save everything
        saccades.to_csv(op.join(target_folder, f'sub-{subject}_ses-{session}_run-{run}_saccades.tsv'), sep='\t', index=False)
        saccades_eyelink.to_csv(op.join(target_folder, f'sub-{subject}_ses-{session}_run-{run}_saccadesel.tsv'), sep='\t', index=False)
        blinks.to_csv(op.join(target_folder, f'sub-{subject}_ses-{session}_run-{run}_blinks.tsv'), sep='\t', index=False)
        d.to_csv(op.join(target_folder, f'sub-{subject}_ses-{session}_run-{run}_pupil.tsv.gz'), sep='\t')


if __name__ == '__main__':
    run_main(main)
