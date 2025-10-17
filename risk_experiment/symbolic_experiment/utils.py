
import re
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def get_subjects(bids_folder='/data/ds-symbolicrisk'):
    bids_folder = Path(bids_folder)

    # Use a raw string to avoid escape sequence warnings
    reg = re.compile(r'.*/sub-(?P<subject>\d+)')

    # Get the list of directories
    dirs = glob.glob(str(bids_folder / 'sourcedata' / 'logs' / 'sub-*'))

    # Extract and sort subject numbers, with error handling
    subjects = sorted(
        reg.match(d).group('subject') for d in dirs if reg.match(d)
    )

    return subjects

# Now `subjects` contains the sorted list of subject numbers


def get_behavioral_data(bids_folder='/data/ds-symbolicrisk', n_risk_bins=3):

    subjects = get_subjects(bids_folder)
    
    df = []
    for subject in subjects:
        d = pd.read_csv(f'/data/ds-symbolicrisk/sourcedata/logs/sub-{subject}/sub-{subject}_task-numeral_gambles_events.tsv', sep='\t', index_col=[0,1,2]).xs('choice', level='event_type')

        df.append(d)

    df = pd.concat(df, keys=subjects, names=['subject'])
    df['risky_first'] = df['prob1'] != 1.0

    df['n_safe'] = df['n1'].where(~df['risky_first'], df['n2'])
    df['n_risky'] = df['n2'].where(~df['risky_first'], df['n1'])

    df['chose_risky'] = (df['choice'] == 1).where(df['risky_first'], df['choice'] == 2)

    df['log(risky/safe)'] = np.log(df['n_risky'] / df['n_safe'])

    bins = np.exp(np.linspace(np.log(5), np.log(28), n_risk_bins + 1))
    lower_bins = bins[:-1]
    lower_bins[0] = 5
    higher_bins = bins[1:]
    higher_bins[-1] = 28
    bin_labels = []
    for i in range(len(lower_bins)):
        bin_labels.append(f'{int(lower_bins[i])}-{int(higher_bins[i])}')
    df['n_safe_bin'] = pd.cut(df['n_safe'], bins=bins, include_lowest=True, labels=bin_labels)

    df['n_safe_bin'] = df['n_safe_bin'].astype(str)
    df['Order'] = df['risky_first'].map({True: 'Risky first', False: 'Safe first'})

    return df


def get_n_safe_bins(n_safe, n_risk_bins=5):
    """
    Returns the bin label for a given number of safe gambles.
    """
    bins = np.exp(np.linspace(np.log(5), np.log(28), n_risk_bins + 1))
    lower_bins = bins[:-1]
    lower_bins[0] = 5
    higher_bins = bins[1:]
    higher_bins[-1] = 28
    bin_labels = []
    for i in range(len(lower_bins)):
        bin_labels.append(f'{int(lower_bins[i])}-{int(higher_bins[i])}')
    
    return pd.cut(n_safe, bins=bins, include_lowest=True, labels=bin_labels)