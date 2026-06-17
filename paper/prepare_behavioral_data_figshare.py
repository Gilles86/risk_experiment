"""
Prepare per-participant behavioral TSV files for figshare upload.

One file per participant, one row per trial, covering both 3T and 7T sessions.
Output: paper/behavioral_data/sub-{id}_behavior.tsv
"""

import os
import os.path as op
import pandas as pd
from risk_experiment.utils.data import get_all_subject_ids, Subject

BIDS_FOLDER = '/data/ds-risk'
OUTPUT_DIR = op.join(op.dirname(__file__), 'behavioral_data')

COLUMNS = [
    'session',
    'run',
    'trial_nr',
    'n1',
    'n2',
    'prob1',
    'prob2',
    'risky_first',
    'n_risky',
    'n_safe',
    'log_risky_safe',
    'choice',
    'chose_risky',
    'rt',
    'certainty',
]

COLUMN_DESCRIPTIONS = {
    'session':        '3t2 = 3T MRI session, 7t2 = 7T MRI session',
    'run':            'Run number within session (1-8)',
    'trial_nr':       'Trial number within run',
    'n1':             'Number of dots in option 1 (stimulus shown first)',
    'n2':             'Number of dots in option 2 (stimulus shown second)',
    'prob1':          'Win probability for option 1 (0.55 = risky, 1.0 = safe)',
    'prob2':          'Win probability for option 2 (0.55 = risky, 1.0 = safe)',
    'risky_first':    'True if the risky option (prob=0.55) was option 1',
    'n_risky':        'Number of dots of the risky option',
    'n_safe':         'Number of dots of the safe option',
    'log_risky_safe': 'log(n_risky / n_safe), the key decision variable',
    'choice':         'Raw response: 1 = chose option 1, 2 = chose option 2',
    'chose_risky':    'True if participant chose the risky option',
    'rt':             'Reaction time in seconds (from onset of option 2 to response)',
    'certainty':      'Post-decision certainty rating (1 = very uncertain, 4 = very certain); stored as "uncertainty" internally',
}


def prepare_subject(subject_id, bids_folder):
    subj = Subject(subject_id, bids_folder)
    df = subj.get_behavior(sessions=['3t2', '7t2'], drop_no_responses=False)

    if df.empty:
        print(f'  sub-{subject_id}: no data found, skipping')
        return None

    df = df.reset_index()

    # Rename columns to TSV-friendly names
    df = df.rename(columns={
        'log(risky/safe)': 'log_risky_safe',
        'uncertainty': 'certainty',
    })

    # Cast booleans to int for cleaner TSV output
    for col in ['risky_first', 'chose_risky']:
        if col in df.columns:
            df[col] = df[col].astype('boolean')

    df['choice'] = df['choice'].astype('Int64')
    df['trial_nr'] = df['trial_nr'].astype('Int64')
    df['run'] = df['run'].astype('Int64')

    return df[COLUMNS]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write a README/codebook alongside the data
    codebook_path = op.join(OUTPUT_DIR, 'codebook.tsv')
    codebook = pd.DataFrame([
        {'column': col, 'description': COLUMN_DESCRIPTIONS[col]}
        for col in COLUMNS
    ])
    codebook.to_csv(codebook_path, sep='\t', index=False)
    print(f'Wrote codebook to {codebook_path}')

    subject_ids = get_all_subject_ids()
    for subject_id in subject_ids:
        print(f'Processing sub-{subject_id}...')
        df = prepare_subject(subject_id, BIDS_FOLDER)
        if df is None:
            continue
        out_path = op.join(OUTPUT_DIR, f'sub-{subject_id}_behavior.tsv')
        df.to_csv(out_path, sep='\t', index=False)
        print(f'  Wrote {len(df)} trials to {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
