from exptools2.core import Session
from exptools2.core import Trial
from session import PileSession
from utils import run_experiment
import numpy as np
from psychopy import logging
import os.path as op
import pandas as pd
from gamble import IntroBlockTrial, GambleTrial, GambleInstructionTrial
from trial import OutroTrial


class CalibrationSession(PileSession):

    Trial = GambleTrial

    def create_trials(self):

        calibrate_settings_folder = op.abspath(
            op.join('settings', 'calibration'))
        trial_settings = pd.read_csv(op.abspath(op.join(calibrate_settings_folder,
                                                        f'sub-{self.subject}_ses-calibrate.tsv')), sep='\t')

        self.n_runs = trial_settings.run.unique().shape[0]

        self.trials = []

        jitter1 = self.settings['calibrate'].get('jitter1')
        jitter2 = self.settings['calibrate'].get('jitter2')

        trial_settings = trial_settings

        for run, d in trial_settings.groupby(['run'], sort=False):
            self.trials.append(GambleInstructionTrial(self, trial_nr=run,
                                                      n_runs=self.n_runs,
                                                      run=run))
            for (p1, p2), d2 in d.groupby(['p1', 'p2'], sort=False):
                n_trials_in_miniblock = len(d2)
                self.trials.append(IntroBlockTrial(session=self, trial_nr=run,
                                                   n_trials=n_trials_in_miniblock,
                                                   prob1=p1,
                                                   prob2=p2))

                for ix, row in d2.iterrows():
                    self.trials.append(GambleTrial(self, row.trial,
                                                   prob1=row.p1, prob2=row.p2,
                                                   num1=int(row.n1),
                                                   num2=int(row.n2),
                                                   jitter1=jitter1,
                                                   jitter2=jitter2))



if __name__ == '__main__':

    session_cls = CalibrationSession
    task = 'calibration'
    run_experiment(session_cls, task=task)
