import os.path as op
from fit_pupil_model import get_data

bids_folder = '/data/ds-risk'

pupil, saccades,  blinks, behavior, events = get_data(None, bids_folder)

pupil.to_parquet(op.join(bids_folder, 'derivatives', 'pupil', 'pupil.parquet'))