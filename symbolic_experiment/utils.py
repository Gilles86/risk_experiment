import os.path as op
import logging
import yaml


def get_output_dir_str(subject, session=None, task='task', run=None):
    output_dir = op.join(op.dirname(__file__), 'logs', f'sub-{subject}')
    logging.warning(f'Writing results to  {output_dir}')

    if session is not None:
        output_dir = op.join(output_dir, f'ses-{session}')
        output_str = f'sub-{subject}_ses-{session}_task-{task}'
    else:
        output_str = f'sub-{subject}_task-{task}'

    if run is not None:
        output_str += f'_run-{run}'

    return output_dir, output_str

def get_settings(settings):
    settings_fn = op.join(op.dirname(__file__), 'settings', f'{settings}.yml')
    print(settings_fn)

    with open(settings_fn, 'r') as f:
        settings = yaml.safe_load(f)

    return settings_fn