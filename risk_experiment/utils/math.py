import numpy as np
from scipy import signal
import pandas as pd
from nilearn import image

def psc(data):
    return image.math_img('(data/data.mean(-1)[..., np.newaxis]) * 100 - 100', data=data)

def resample_run(d, new_samplerate=10, old_samplerate=500):
    new_n = int(np.ceil(len(d) / old_samplerate * new_samplerate))
    resampled_pupil = pd.DataFrame(signal.resample(d['pupil'], new_n), columns=['pupil'])
    resampled_pupil['time'] = np.linspace(0, new_n*(1./new_samplerate), new_n, False)
    return resampled_pupil